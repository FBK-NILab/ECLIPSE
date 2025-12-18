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

def step_size_streamline(streamline):
    """
    streamline: array (N, 3)
    return: average step-size
    """
    diffs = np.diff(streamline, axis=0)      # (N-1, 3)
    dists = np.linalg.norm(diffs, axis=1)    # (N-1,)
    return dists.mean()


def get_results(config, rank):
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

    K = config.get('Top_k_error', 10)
    K = max(min(K, 10), 1)
    data_config = config['data']

    val = ''
    if config["subsample"]:
        val = 'subset_'

    # Done to save the compilation time when the loading function is actually called
    precompile_numba_functions()

    test_sub_list = open(f'src/splits/{val}split_test.txt', 'r').readlines()
    test_sub_list = [item.strip() for item in test_sub_list]

    data_config['sub_list'] = test_sub_list
    data_config['mode'] = 'test'
    test_data = get_dataset(data_config)
    # test_data.set_streamline_number(2000)
    
    tester = Evaluator(config, test_data, curr_seed=seed)
    lines, embeddings, errors, lengths, sub_names, reconstructions = tester.embedding_distance_and_error(geometric)
    reset_all_loggers()
    
    return {"lines": lines, "embeddings": embeddings, "errors": errors, "lengths": lengths,
            "sub_names": sub_names, "reconstructions": reconstructions}


def compute_stats(lines, emb, error, length, name, recons, geometric):
    if geometric:
        sub_lines = [torch.stack(l).numpy() for l in lines]
        sub_errors = [torch.stack(e).numpy() for e in error]
        sub_reconstructions = [torch.stack(r).numpy() for r in recons]
    else:
        if not args.with_pad_data:
            sub_lines = [l[:leng].numpy() for leng, l in zip(length, lines)]
            sub_errors = [e.numpy() for e in error]
            sub_reconstructions = [r[:leng].numpy() for leng, r in zip(length, recons)]
        else:
            sub_lines = [l.numpy() for leng, l in lines]
            sub_errors = [e.numpy() for e in error]
            sub_reconstructions = [r.numpy() for leng, r in recons]


    avg_streamline_error = [np.sum(error) / (elem * 3) for elem, error in zip(length, sub_errors)]
    return sub_lines, sub_errors, sub_reconstructions, avg_streamline_error


def main(rank) -> None:

    mpl.use('Agg')

    res_finta = get_results(config_finta, rank)
    res_eclipse = get_results(config_gnn_eclipse, rank)
    res_dyn =get_results(config_gnn_dyn, rank)

    for idx in range(len(res_finta["lines"])):

        lines_finta, emb_finta, error_finta, length_finta, name_finta, recons_finta = [res_finta[k][idx] for k in res_finta.keys()]
        lines_eclipse, emb_eclipse, error_eclipse, length_eclipse, name_eclipse, recons_eclipse = [res_eclipse[k][idx] for k in
                                                                                       res_eclipse.keys()]
        lines_dyn, emb_dyn, error_dyn, length_dyn, name_dyn, recons_dyn = [res_dyn[k][idx] for k in
                                                                                       res_dyn.keys()]

        K = 20
        sub_imgs_path = os.path.join(f"imgs/recon_analys_1mm_haus_marker_{args.with_marker}_baseline_{args.baseline}",
                                     name_finta)
        os.makedirs(sub_imgs_path, exist_ok=True)

        sub_lines_finta, sub_errors_finta, sub_reconstructions_finta, avg_streamline_error_finta = compute_stats(lines_finta, emb_finta, error_finta, length_finta, name_finta, recons_finta, False)
        sub_lines_eclipse, sub_errors_eclipse, sub_reconstructions_eclipse, avg_streamline_error_eclipse = compute_stats(lines_eclipse, emb_eclipse, error_eclipse, length_eclipse, name_eclipse, recons_eclipse, True)
        sub_lines_dyn, sub_errors_dyn, sub_reconstructions_dyn, avg_streamline_error_dyn = compute_stats(lines_dyn, emb_dyn, error_dyn, length_dyn, name_dyn, recons_dyn, True)
        
        if args.baseline == 'FINTA':
            top_k_idx = np.argsort(avg_streamline_error_finta)[::-1][:K]
        else:
            top_k_idx = np.argsort(avg_streamline_error_eclipse)[::-1][:K]

        top_lines_finta = [sub_lines_finta[i] for i in top_k_idx]
        top_recons_finta = [sub_reconstructions_finta[i] for i in top_k_idx]
        top_errors_finta = [avg_streamline_error_finta[i] for i in top_k_idx]

        top_lines_eclipse = [sub_lines_eclipse[i] for i in top_k_idx]
        top_recons_eclipse = [sub_reconstructions_eclipse[i] for i in top_k_idx]
        top_errors_eclipse = [avg_streamline_error_eclipse[i] for i in top_k_idx]

        top_lines_dyn = [sub_lines_dyn[i] for i in top_k_idx]
        top_recons_dyn = [sub_reconstructions_dyn[i] for i in top_k_idx]
        top_errors_dyn = [avg_streamline_error_dyn[i] for i in top_k_idx]


        for j in range(len(top_lines_finta)):
            fig = plt.figure()
            gt = top_lines_finta[j]
            finta_recon = top_recons_finta[j]
            eclipse_recon = top_recons_eclipse[j]
            dyn_recon = top_recons_dyn[j]
            print(
                f"img={sub_imgs_path} ste_size={step_size_streamline(gt)} streamline={top_k_idx[j]} plot_num={j} error_finta={len(finta_recon)} error_eclipse={len(eclipse_recon)}, error_dyn={len(dyn_recon)}")

            ax = fig.add_subplot(111, projection='3d')
            if args.with_marker:
                ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="blue", marker='x', linewidth=1, label="gt")
                # reconstruction
                ax.plot(finta_recon[:, 0], finta_recon[:, 1], finta_recon[:, 2], marker='o', markersize=1, color="red",
                        linewidth=1, label="finta")
                ax.plot(eclipse_recon[:, 0], eclipse_recon[:, 1], eclipse_recon[:, 2], color="black", marker='^',
                        markersize=1, linewidth=1, label="ECLIPSE")
            else:
                # original streamline
                ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="blue", label="gt")
                # reconstruction
                ax.plot(finta_recon[:, 0], finta_recon[:, 1], finta_recon[:, 2], color="red", label="finta")
                ax.plot(eclipse_recon[:, 0], eclipse_recon[:, 1], eclipse_recon[:, 2], color="black", label="eclipse")

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            fig.legend(loc=7)
            fig.tight_layout()
            fig.subplots_adjust(right=0.75)
            fig.suptitle(f"Avg error ECLIPSE={top_errors_eclipse[j]:.2f}, FINTA={top_errors_finta[j]:.2f} \n"
                         f"Avg error chamfer ECLIPSE={chamfer_mdf(eclipse_recon, gt):.2f}, FINTA={chamfer_mdf(finta_recon, gt):.2f}")

            plt.savefig(os.path.join(sub_imgs_path, f'streamline_{j}_error.png'), format='png')
            plt.savefig(os.path.join(sub_imgs_path, f'streamline_{j}_error.pdf'), format='pdf')
            plt.savefig(os.path.join(sub_imgs_path, f'streamline_{j}_error.svg'), format='svg')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_pad_data', '-p', help='Add pad data in plot', default=False, required=False, action='store_true')
    parser.add_argument('--with_marker', '-m', help='Add point marker', default=False, required=False,
                        action='store_true')
    parser.add_argument('--baseline', help="Baseline for top k error", default='FINTA', required=False, type=str)
    args = parser.parse_args()

    config_finta = yaml.safe_load(open('src/configs/config_finta_padding.yaml', 'r'))
    config_gnn_eclipse = yaml.safe_load(open('src/configs/template_gnn_autoencoder_test.yaml', 'r'))
    config_gnn_dyn = yaml.safe_load(open('src/configs/template_gnn_autoencoder_test.yaml', 'r'))

    config_finta = translate_config(config_finta)
    config_gnn_eclipse = translate_config(config_gnn_eclipse)
    config_gnn_dyn = translate_config(config_gnn_dyn)

    config_finta.update({"name": 'train_FINTA_padding_new_mask', 'y': True, 'subsample': True})
    config_gnn_eclipse.update({"name": 'train_gnn_autoencoder_no_dyn', 'y': True, 'subsample': True})
    config_gnn_dyn.update({"name": 'train_gnn_autoencoder_no_dyn', 'y': True, 'subsample': True})



    """
    config_finta = yaml.safe_load(open('src/configs/config_finta_padding_new_test.yaml', 'r'))
    config_gnn_no_dyn = yaml.safe_load(open('src/configs/config_gnn_autoencoder_no_dyn_test.yaml', 'r'))
    config_gnn_dyn = yaml.safe_load(open('src/configs/config_gnn_autoencoder_adamw_steprl_fast_slow_relu_test.yaml', 'r'))

    config_finta = translate_config(config_finta)
    config_gnn_no_dyn = translate_config(config_gnn_no_dyn)
    config_gnn_dyn = translate_config(config_gnn_dyn)

    config_finta.update({"name": 'train_FINTA_padding_new_mask', 'y': True, 'subsample': False})
    config_gnn_no_dyn.update({"name": 'train_gnn_autoencoder_no_dyn', 'y': True, 'subsample': False})
    config_gnn_dyn.update({"name": 'train_gnn_steprl_adamw_fast_slow_relu', 'y': True, 'subsample': False})
    """
    config_finta['loss']['reduction'] = 'none'
    config_finta['mode'] = 'test'

    config_gnn_eclipse['loss']['reduction'] = 'none'
    config_gnn_eclipse['mode'] = 'test'

    config_gnn_dyn['loss']['reduction'] = 'none'
    config_gnn_dyn['mode'] = 'test'

    init_cudnn(deterministic=config_finta["deterministic"], benchmark=config_finta["benchmark"])

    main(rank=int(os.getenv('LOCAL_RANK', 0)))
