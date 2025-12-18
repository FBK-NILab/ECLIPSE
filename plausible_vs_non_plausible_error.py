import yaml
from pprintpp import pprint
from torch.distributed import destroy_process_group
import torch.multiprocessing as mp
import copy

from src import *


def main(rank):
    setup_distributed_training(config, rank)
    checkpoint_path = config.get('checkpoint_path', 'checkpoints')
    experiment_name = config["name"]
    checkpoint_path = os.path.join(checkpoint_path, experiment_name)
    config['checkpoint_path'] = checkpoint_path
    imgs_path = os.path.join(config.get('imgs_path', 'imgs'), experiment_name)
    config['imgs_path'] = imgs_path
    if config.get('fixed_seed'):
        # To ensure different but deterministic seeds across ranks, offset by rank
        seed = config.get('seed', 0) + rank if config.get('ranked_seed') else config.get('seed')
        set_random_seed(seed)
    else:
        seed = torch.random.initial_seed() #we need this for ddp, but usually it is BAD to not set a fixed seed

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
    test_data.set_plausible_only(True)
    test_data.set_unplausible_only(False)
    config_new = copy.deepcopy(config)
    tester = Evaluator(config, test_data, curr_seed=seed)
    tester.test(f"=== Test (Only plausible) {config.get('which_epoch')}===")

    test_data.set_plausible_only(False)
    test_data.set_unplausible_only(True)
    reset_all_loggers()
    tester = Evaluator(config_new, test_data, curr_seed=seed)
    tester.test(f"=== Test (Only NON plausible) {config.get('which_epoch')}===")

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
