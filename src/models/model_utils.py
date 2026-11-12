import torch
from torch.nn import LeakyReLU, ReLU, GELU

from .ECLIPSE import ECLIPSE



_all_activations = [LeakyReLU, ReLU, GELU]
_available_activations = [item.__name__.lower() for item in _all_activations]

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


def get_model(model_config:dict) -> torch.nn.Module:

    activation_name = model_config.pop('activation', 'relu')
    activation = None if activation_name == '' else get_activation(activation_name)
    model_config['activation'] = activation
    model = ECLIPSE(
        **model_config
    )

    return model

