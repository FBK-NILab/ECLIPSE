from datetime import timedelta
import os
import functools

import torch.distributed as dist
import torch


def get_world_size():
    r"""Get world size. How many GPUs are available in this job."""
    world_size = 1
    if dist.is_available():
        if dist.is_initialized():
            world_size = dist.get_world_size()
    return world_size


def set_cuda_visible_devices(device_ids: list):
    """
    Set CUDA visible device ids.
    Args:
        device_ids: list of CUDA visible device ids.

    Returns:

    """
    os.environ['CUDA_VISIBLE_DEVICES'] = ", ".join([str(device_id) for device_id in device_ids])


def master_only(func):
    r"""Apply this function only to the master GPU."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        r"""Simple function wrapper for the master function"""
        if get_rank() == 0:
            return func(*args, **kwargs)
        else:
            return None
    return wrapper


def is_local_master():
    return torch.cuda.current_device() == 0


@master_only
def master_only_print(*args):
    r"""master-only print"""
    print(*args)


def dist_all_reduce_tensor(tensor, reduce="mean"):
    r""" Reduce to all ranks """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.all_reduce(tensor)
        if reduce == 'mean':
            tensor /= world_size
        elif reduce == 'sum':
            pass
        else:
            raise NotImplementedError
    return tensor


def dist_all_gather_tensor(tensor):
    r""" gather to all ranks """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor_list = [
        torch.ones_like(tensor, device=get_rank()) for _ in range(dist.get_world_size())]
    with torch.no_grad():
        dist.all_gather(tensor_list, tensor)

    return torch.cat(tensor_list, dim=0)


def is_master():
    """
        check if current process is the master
    Returns:

    """
    return get_rank() == 0


def get_rank():
    """
    get current rank
    Returns:
    """
    rank = 0
    if dist.is_available():
        if dist.is_initialized():
            rank = dist.get_rank()
    return rank


def setup_distributed_training(config, rank):
    if len(config["device_ids"]) >= 1:
        torch.cuda.set_device(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        if os.name == 'nt':  # Windows
            torch.distributed.init_process_group(backend='gloo', rank=rank,
                                                 world_size=len(config["device_ids"]))
        else:  # Unix
            os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '0'  # eval is performed on a single GPU, this avoids timeout from other waiting processes
            torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank,
                                                 world_size=len(config["device_ids"]),
                                                 timeout=timedelta(seconds=10000))
        print(
            f"[{os.getpid()}] world_size = {get_world_size()}, "
            + f"rank = {get_rank()}, backend={dist.get_backend()}"
        )
