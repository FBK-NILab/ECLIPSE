import os.path

import yaml
from pprintpp import pprint
from torch.distributed import destroy_process_group
import torch.multiprocessing as mp

from src import *


def main(rank):
    setup_distributed_training(config, rank)
    mode = config['mode']

    assert mode in ['train', 'test'], 'Please make sure that the mode is either "train" or "test"'

    checkpoint_path = config.get('checkpoint_path', 'checkpoints')
    experiment_name = config["name"]
    if experiment_name is None:
        experiment_name = 'experiment_'
        last_exp = -1
        if os.path.exists(checkpoint_path):
            for folder in os.listdir(checkpoint_path):
                if 'experiment_' in folder:
                    exp = int(folder.split('_')[-1])
                    if exp > last_exp:
                        last_exp = exp

        experiment_name += str(last_exp + 1)

    checkpoint_path = os.path.join(checkpoint_path, experiment_name)
    if is_master() and mode == 'train' and not config.get("continue_train", False):
        mkdirs(checkpoint_path, config["y"])

    config['checkpoint_path'] = checkpoint_path

    imgs_path = os.path.join(config.get('imgs_path', 'imgs'), experiment_name)
    config['imgs_path'] = imgs_path
    if is_master() and mode == 'train' and not config.get("continue_train", False):
        mkdirs(imgs_path, config["y"])

    if config.get('fixed_seed'):
        # To ensure different but deterministic seeds across ranks, offset by rank
        seed = config.get('seed', 0) + rank if config.get('ranked_seed') else config.get('seed')
        set_random_seed(seed)
    else:
        seed = torch.random.initial_seed() #we need this for ddp, but usually it is BAD to not set a fixed seed

    data_config = config['data']
    data_config['mode'] = mode

    val = ''
    if config["subsample"]:
        val = 'subset_'

    # Done to save the compilation time when the loading function is actually called
    precompile_numba_functions()

    if mode == 'train':
        train_sub_list = open(f'src/splits/{val}split_train.txt', 'r').readlines()
        val_sub_list = open(f'src/splits/{val}split_val.txt', 'r').readlines()
        test_sub_list = open(f'src/splits/{val}split_test.txt', 'r').readlines()
        train_sub_list = [item.strip() for item in train_sub_list]
        val_sub_list = [item.strip() for item in val_sub_list]

        data_config['sub_list'] = train_sub_list
        data_config['mode'] = 'train'
        train_data = get_dataset(data_config)

        data_config['sub_list'] = val_sub_list
        data_config['mode'] = 'val'
        val_data = get_dataset(data_config)

        data_config['sub_list'] = test_sub_list
        data_config['mode'] = 'test'
        test_data = get_dataset(data_config)

        trainer = Trainer(config, train_data, val_data, test_data, seed)
        result_train = trainer.train()
        if is_master():
            trainer.test(f"=== Test plausible_only={config.get('plausible_only', True)}" 
                         f" non_plausible_only={config.get('non_plausible_only', False)}"
                         f"epoch {trainer.current_epoch}===")
            trainer.model.load_state_dict({k: v.to(get_rank()) for k, v in result_train["best_state_model"].items()})
            for k, v in trainer.test_dict.items():
                trainer.test_dict[k].reset()
            trainer.test(f"=== Test plausible_only={config.get('plausible_only', True)}" 
                         f" non_plausible_only={config.get('non_plausible_only', False)}"
                         f" best epoch===")
    else:
        test_sub_list = open(f'src/splits/{val}split_test.txt', 'r').readlines()
        test_sub_list = [item.strip() for item in test_sub_list]
        data_config['sub_list'] = test_sub_list
        data_config['mode'] = 'test'
        test_data = get_dataset(data_config)
        tester = Evaluator(config, test_data, curr_seed=seed)
        tester.test(f"=== Test plausible_only={config.get('plausible_only', True)}"
                    f" non_plausible_only={config.get('non_plausible_only', False)}"
                    f" (epoch {config.get('which_epoch', None)}) ===")

    destroy_process_group()


if __name__ == '__main__':

    args = Parameters().parse()
    assert os.path.exists(args.config_path), f'The config path ({args.config_path}) does not exist'

    config = yaml.safe_load(open(args.config_path, 'r'))
    config = translate_config(config)
    config.update(vars(args))
    pprint(config)
    print('=' * 100, '\n')
    init_cudnn(deterministic=config["deterministic"], benchmark=config["benchmark"])
    if len(config["device_ids"]) > 1:
        set_cuda_visible_devices(config["device_ids"])
        mp.spawn(main, nprocs=len(config["device_ids"]))
    else:
        main(rank=int(os.getenv('LOCAL_RANK', 0)))

