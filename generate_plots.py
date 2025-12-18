import os
import random
from itertools import combinations


import yaml
from pprint import pprint
from dipy.tracking.streamline import length as streamline_length_mm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src import *


def main(rank) -> None:

    mpl.use('Agg')

    if config.get('fixed_seed'):
        # To ensure different but deterministic seeds across ranks, offset by rank
        seed = config.get('seed', 0) + rank if config.get('ranked_seed') else config.get('seed')
        set_random_seed(seed)
    else:
        seed = torch.random.initial_seed() # we need this for ddp, but usually it is BAD to not set a fixed seed

    img_path = config.get('img_path', 'imgs')
    experiment_name = config["name"]
    if experiment_name is None:
        experiment_name = 'experiment_'
        last_exp = -1
        if os.path.exists(img_path):
            for folder in os.listdir(img_path):
                if 'experiment_' in folder:
                    exp = int(folder.split('_')[-1])
                    if exp > last_exp:
                        last_exp = exp

        experiment_name += str(last_exp + 1)

    img_path = os.path.join(img_path, experiment_name, config['model']['model_name'])
    os.makedirs(img_path, exist_ok=True)

    checkpoint_path = config.get('checkpoint_path', 'checkpoints')
    experiment_name = config["name"]
    checkpoint_path = os.path.join(checkpoint_path, experiment_name)
    config['checkpoint_path'] = checkpoint_path

    geometric = config['data']['return_geometric']
    use_chamfer = config.get('use_chamfer_distance', False)

    N = config.get('error_vs_length_test_streamline_number', 2000)  # Number of streamlines to sample in avg error vs lenth analysis
    M = config.get('embedding_vs_real_distance_test_streamline_number', 100)  # Number of streamlines to sample in embedding vs real distanca analysis
    idxs = range(M)
    couple_idxs = list(combinations(idxs, 2))
    random.shuffle(couple_idxs)

    assert M <= N, f'The number of streamlines of the error vs length test must at least equal to the number of embedding vs real distance test. Found {N} <= {M}'

    data_config = config['data']

    val = ''
    if config["subsample"]:
        val = 'subset_'

    cmap_name = config.get('cmap_name', 'viridis')

    # Done to save the compilation time when the loading function is actually called
    precompile_numba_functions()

    test_sub_list = open(f'src/splits/{val}split_test.txt', 'r').readlines()
    test_sub_list = [item.strip() for item in test_sub_list]

    data_config['sub_list'] = test_sub_list
    data_config['mode'] = 'test'
    test_data = get_dataset(data_config)

    test_data.set_streamline_number(N)

    tester = Evaluator(config, test_data, curr_seed=seed)
    lines, embeddings, errors, lengths, sub_names, _ = tester.embedding_distance_and_error(geometric)


    for l, line, embedding, error, name in zip(lengths, lines, embeddings, errors, sub_names):

        if geometric:
            sub_lines = [torch.stack(l).numpy() for l in line]
            sub_errors = [torch.stack(e).numpy() for e in error]
        else:
            sub_lines = [l.numpy() for l in line]
            sub_errors = [e.numpy() for e in error]

        embedding = torch.stack(embedding).numpy()
        sub_imgs_path = os.path.join(img_path, name)
        os.makedirs(sub_imgs_path, exist_ok=True)

        mm_lengths = [streamline_length_mm(line) for line in sub_lines]
        avg_streamline_error = [np.sum(error) / (elem * 3) for elem, error in zip(l, sub_errors)]
        sum_streamline_error = [np.sum(error) for error in sub_errors]


        # Graph of reconstruction_error vs length
        fig, axes = plt.subplots(1, figsize=(8, 12))
        axes.scatter(mm_lengths, avg_streamline_error)

        axes.set_title("Length vs avg error on the streamline")
        axes.set_xlabel("Streamline length (mm)")
        axes.set_ylabel("Average streamline error (MSELoss)")
        axes.set_ylim(0, 200)
        axes.set_yticks([x for x in range(0,201, 50)])

        plt.tight_layout()
        plt.savefig(os.path.join(sub_imgs_path, f'Streamline_length_vs_error.png'), format='png')
        plt.savefig(os.path.join(sub_imgs_path, f'Streamline_length_vs_error.pdf'), format='pdf')
        plt.savefig(os.path.join(sub_imgs_path, f'Streamline_length_vs_error.svg'), format='svg')
        plt.close()

        # Graph embedding vs real distance
        first_idxs = [couple[0] for couple in couple_idxs]
        second_idxs = [couple[1] for couple in couple_idxs]

        first_lines = [sub_lines[idx] for idx in first_idxs]
        second_lines = [sub_lines[idx] for idx in second_idxs]

        len_diffs = [abs(streamline_length_mm(first_line) - streamline_length_mm(second_line)) for first_line, second_line in zip(first_lines, second_lines)]

        if use_chamfer:
            real_dists_mdf = [chamfer_mdf(first_line, second_line) for first_line, second_line in zip(first_lines, second_lines)]
        else:
            real_dists_mdf = [hausdorff_mdf(first_line, second_line) for first_line, second_line in zip(first_lines, second_lines)]

        first_embeddings = [embedding[idx] for idx in first_idxs]
        second_embeddings = [embedding[idx] for idx in second_idxs]
        emb_dists = [np.linalg.norm(first_emb - second_emb) for first_emb, second_emb in zip(first_embeddings, second_embeddings)]
        emb_dists_cosine = [np.dot(first_emb, second_emb) / (np.linalg.norm(first_emb) * np.linalg.norm(second_emb)) for first_emb, second_emb in zip(first_embeddings, second_embeddings)]


        x_coords_fn = f"{config['model']['model_name']}_{experiment_name}_embedding_dists.npy"
        x_coords_fn_cos = f"{config['model']['model_name']}_{experiment_name}_embedding_dists_cosine.npy"
        y_coords_fn = f"{config['model']['model_name']}_{experiment_name}_mdf_dists.npy"

        np.save(os.path.join(sub_imgs_path, x_coords_fn), emb_dists)
        np.save(os.path.join(sub_imgs_path, y_coords_fn), real_dists_mdf)
        np.save(os.path.join(sub_imgs_path, x_coords_fn_cos), emb_dists_cosine)
        np.save(os.path.join(sub_imgs_path, 'first_embeddings.npy'), np.array(first_embeddings))
        np.save(os.path.join(sub_imgs_path, 'second_embeddings.npy'), np.array(second_embeddings))

        fig = plt.figure(figsize=(8, 12))
        gs = gridspec.GridSpec(
            1, 2,  # 1 row, 2 columns
            width_ratios=[1, 0.05],  # last column reserved for colorbar
            wspace=0.25
        )

        ax0 = fig.add_subplot(gs[0, 0])
        cbar_ax = fig.add_subplot(gs[0, 1])  # axis dedicated to colorbar

        sc0 = ax0.scatter(
            emb_dists,
            real_dists_mdf,
            c=len_diffs,
            cmap=cmap_name
        )

        ax0.set_title("MDF vs embedding distances")
        ax0.set_xlabel("Embedding distance (euclidean)")
        ax0.set_ylabel("MDF distance")

        make_axes_locatable(ax0)
        make_axes_locatable(cbar_ax)

        cbar = fig.colorbar(sc0, cax=cbar_ax, orientation='vertical', location='right')
        cbar.set_label("Length difference", rotation=270, labelpad=15)

        plt.tight_layout()
        plt.savefig(os.path.join(sub_imgs_path, f'Embedding_vs_real_distance.png'), format='png')
        plt.savefig(os.path.join(sub_imgs_path, f'Embedding_vs_real_distance.pdf'), format='pdf')
        plt.savefig(os.path.join(sub_imgs_path, f'Embedding_vs_real_distance.svg'), format='svg')
        plt.close()


        fig = plt.figure(figsize=(8, 12))
        gs = gridspec.GridSpec(
            1, 2,  # 1 row, 2 columns
            width_ratios=[1, 0.05],  # last column reserved for colorbar
            wspace=0.25
        )

        ax0 = fig.add_subplot(gs[0, 0])
        cbar_ax = fig.add_subplot(gs[0, 1])  # axis dedicated to colorbar

        sc0 = ax0.scatter(
            emb_dists_cosine,
            real_dists_mdf,
            c=len_diffs,
            cmap=cmap_name
        )
        ax0.set_title("MDF vs embedding distances")
        ax0.set_xlabel("Cosine similarity")
        ax0.set_ylabel("MDF distance")

        make_axes_locatable(ax0)
        make_axes_locatable(cbar_ax)

        cbar = fig.colorbar(sc0, cax=cbar_ax, orientation='vertical', location='right')
        cbar.set_label("Length difference", rotation=270, labelpad=15)

        plt.tight_layout()
        plt.savefig(os.path.join(sub_imgs_path, f'Embedding_vs_real_distance_cosine.png'), format='png')
        plt.savefig(os.path.join(sub_imgs_path, f'Embedding_vs_real_distance_cosine.pdf'), format='pdf')
        plt.savefig(os.path.join(sub_imgs_path, f'Embedding_vs_real_distance_cosine.svg'), format='svg')
        plt.close()


if __name__ == '__main__':

    args = Parameters().parse()
    assert os.path.exists(args.config_path), f'The config path ({args.config_path}) does not exist'

    config = yaml.safe_load(open(args.config_path, 'r'))
    config = translate_config(config)
    config.update(vars(args))
    config['loss']['reduction'] = 'none'

    pprint(config)
    print('=' * 100, '\n')
    init_cudnn(deterministic=config["deterministic"], benchmark=config["benchmark"])

    main(rank=int(os.getenv('LOCAL_RANK', 0)))
