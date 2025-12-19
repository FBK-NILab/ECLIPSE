import copy
from typing import Optional, Tuple, List

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LRScheduler
from torch.optim import Adam, Adadelta, Adagrad, SGD, RMSprop, AdamW
from torch.nn import LeakyReLU, ReLU, GELU

from src.utils import get_rank
from .FINTA import FINTA
from .ECLIPSE import ECLIPSE


_all_optim = [
    Adam,
    AdamW,
    Adadelta,
    Adagrad,
    SGD,
    RMSprop
]

_all_lr_scheduler = [
    CosineAnnealingLR,
    StepLR
]

_all_activations = [LeakyReLU, ReLU, GELU]

_available_optimizers = [item.__name__.lower() for item in _all_optim]
_available_scheduler = [item.__name__.lower() for item in _all_lr_scheduler]
_available_activations = [item.__name__.lower() for item in _all_activations]

_valid_names = ['FINTA', 'ECLIPSE']


def get_activation(activation_name: str) -> torch.nn.Module:
    activation = None
    
    for item in _all_activations:
        if item.__name__.lower() == activation_name.lower():
            if activation_name.lower() == 'leakyrelu':
                activation = item(0.2)
            else:
                activation = item()
    
    assert activation is not None, (
        f'The activation chosen ({activation_name}) in the configuration file is not supported yet,'
        f' please choose one among {_available_activations}'
    )
    return activation


def get_optimizer(network: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    optimizer = None
    optimizer_name = config.pop('name', 'adam')

    if optimizer_name.lower() == 'adam' or optimizer_name.lower() == 'adamw':
        if 'betas' in config:
            config['betas'] = (config['betas'].replace('(', '').replace(')', '')
                                       .strip().split(','))
            config['betas'] = [float(item) for item in config['betas']]
            config['betas'] = tuple(config['betas'])

    for item in _all_optim:
        if item.__name__.lower() == optimizer_name.lower():
            # If model is wrapped by DDP, use .module to get underlying parameters
            target_network = getattr(network, 'module', network)
            parameter_list = list(target_network.parameters())
            optimizer = item(parameter_list, **config)

    assert optimizer is not None, (
        f'The optimizer chosen ({optimizer_name}) in the configuration file is not supported yet,'
        f' please choose one among {_available_optimizers}'
    )

    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> Optional[LRScheduler]:

    scheduler_name = config.get('name', '')
    scheduler_config = config.get(scheduler_name.lower(), {})
    
    if scheduler_name.lower() == 'step':
        return StepLR(optimizer, **scheduler_config)
    elif scheduler_name.lower() == 'cosine':
        return CosineAnnealingLR(optimizer, **scheduler_config)
    elif scheduler_name == '':
        return None
    else:
        raise NotImplementedError(
            f'The scheduler {scheduler_name} is not implemented yet. please choose one among {_available_scheduler}')


def get_model(cfg:dict) -> Tuple[torch.nn.Module, Optional[List]]:

    model_name = cfg['model_name']
    model_config = copy.deepcopy(cfg[model_name.lower()])
    
    if model_name == 'FINTA':
        model = FINTA(
            **model_config
        ).to(get_rank())
        input_shape = (1, 10000, 256, 3)
    elif model_name == 'ECLIPSE':
        activation_name = model_config.pop('activation', 'relu')
        activation = None if activation_name == '' else get_activation(activation_name)
        model = ECLIPSE(
            activation=activation,
            **model_config
        ).to(get_rank())
        input_shape = None
    else:
        raise NotImplementedError(f'The dataset {model_name} is not implemented yet.'
                                  f' Please choose among {_valid_names}')

    return model, input_shape


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]['lr']

