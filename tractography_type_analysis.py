import os

import yaml
from pprint import pprint
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
        seed = torch.random.initial_seed()  # we need this for ddp, but usually it is BAD to not set a fixed seed

    data_config = config['data']

    # Done to save the compilation time when the loading function is actually called
    precompile_numba_functions()

    data_config['mode'] = 'test'
    test_data = get_dataset(data_config)
    config_sur = copy.deepcopy(config)
    config_prob = copy.deepcopy(config)
    config_filtered = copy.deepcopy(config)
    
    test_data.set_pattern('det')
    tester = Evaluator(config, test_data, curr_seed=seed)
    tester.test(f"=== Test (TractoInferno deterministic tractography) {config.get('which_epoch')}===")
    
    reset_all_loggers()    
    test_data.set_pattern('prob')
    tester = Evaluator(config_prob, test_data, curr_seed=seed)
    tester.test(f"=== Test (TractoInferno probabilistic tractography) {config.get('which_epoch')}===")
    reset_all_loggers()
    
    test_data.set_pattern('pft')    
    tester = Evaluator(config_filtered, test_data, curr_seed=seed)
    tester.test(f"=== Test (TractoInferno particle-filtered tractography) {config.get('which_epoch')}===")
    reset_all_loggers()
    
    test_data.set_pattern('set')
    tester = Evaluator(config_sur, test_data, curr_seed=seed)
    tester.test(f"=== Test (TractoInferno surface-enhanced tractography) {config.get('which_epoch')}===")



if __name__ == '__main__':

    args = Parameters().parse()
    assert os.path.exists(args.config_path), f'The config path ({args.config_path}) does not exist'

    config = yaml.safe_load(open(args.config_path, 'r'))
    config = translate_config(config)
    config.update(vars(args))

    pprint(config)
    print('=' * 100, '\n')
    init_cudnn(deterministic=config["deterministic"], benchmark=config["benchmark"])

    main(rank=int(os.getenv('LOCAL_RANK', 0)))

