import warnings
from typing import Tuple, Union
import torch


_all_losses = [
    torch.nn.MSELoss,
    torch.nn.L1Loss,
    torch.nn.NLLLoss
]

_available_losses = [item.__name__.lower() for item in _all_losses]


def get_loss(config: dict) -> Tuple[torch.nn.Module, str]:

    loss = None
    loss_name = config.pop('name', '')

    for item in _all_losses:
        if item.__name__.lower() == loss_name.lower():

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                loss = item(**config)

    assert loss is not None, ('The loss chosen in the configuration file is not supported yet,'
                              f' please choose one among {_available_losses}')

    return loss, loss_name
