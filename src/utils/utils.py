import os
import random
import shutil
import logging

from numba import njit, prange
from scipy.spatial import KDTree
import numpy as np
from numpy.typing import NDArray
import nibabel as nib

import torch
import torch.backends.cudnn as cudnn


@njit(parallel=False, fastmath=True, cache=True)
def parse_lengths_and_offsets(buffer_int32, nb_streamlines, point_size, n_properties):
    lengths = np.empty(nb_streamlines, np.int32)
    offsets = np.empty(nb_streamlines, np.int64)

    pointer = 0
    for i in range(nb_streamlines):
        l = buffer_int32[pointer]
        lengths[i] = l
        offsets[i] = pointer
        pointer += 1 + l * point_size + n_properties

    return lengths, offsets


@njit(parallel=True, fastmath=True, cache=True)
def apply_affine_to_streamlines(streamlines, rotation, translation):

    out = []

    rotT = np.ascontiguousarray(rotation.T)
    transT = np.ascontiguousarray(translation.T)

    for i in prange(len(streamlines)):
        s = streamlines[i]
        s_c = np.ascontiguousarray(s)
        s_aff = s_c @ rotT + transT
        out.append(s_aff)
    return out


@njit(fastmath=True, cache=True)
def compute_offsets_from_lengths(lengths, point_size, n_properties):

    nb_streamlines = len(lengths)
    offsets = np.empty(nb_streamlines, np.int64)
    pos = 0
    for i in range(nb_streamlines):
        offsets[i] = pos
        pos += 1 + lengths[i] * point_size + n_properties
    return offsets



@njit(parallel=True, fastmath=True, cache=True)
def apply_affine_to_streamlines_batch(arr, rotation, translation):
    """
    arr: float32 (N, L, 3)
    rotation: float32 (3,3)
    translation: float32 (3,1) or (3,)
    returns out: float32 (N, L, 3)
    """
    rotT = rotation.T
    trans = translation.ravel()  # ravel translation for faster broadcast
    out = np.empty_like(arr)
    for i in prange(arr.shape[0]):
        for j in range(arr.shape[1]):
            out[i, j] = arr[i, j] @ rotT + trans
    return out


def precompile_numba_functions():

    parse_lengths_and_offsets.compile("(int32[:], int64, int64, int64)")
    compute_offsets_from_lengths.compile("(int32[:], int64, int64)")
    apply_affine_to_streamlines_batch.compile("(float32[:, :, :], float32[:, :], float32[:, :])")


def load_streamlines(trk_fn, idxs=None, apply_affine=True, container='list',
                     replace=False, return_len=True, plausible_only=False, non_plausible_only=False, labels=None, mode='train'):

    assert not (plausible_only and non_plausible_only), ("You can only specify either plausible only OR non plausible only, not both at the same time. "
                                                         "If you want to use all streamlines, please set them both to false")

    lazy_trk = nib.streamlines.load(trk_fn, lazy_load=True)
    header = lazy_trk.header
    nb_streamlines = header['nb_streamlines']
    n_scalars = header['nb_scalars_per_point']
    n_properties = header['nb_properties_per_streamline']
    aff = nib.streamlines.trk.get_affine_trackvis_to_rasmm(header)

    rotation = aff[:3, :3].astype(np.float32)
    translation = aff[:3, 3:4].astype(np.float32)

    point_size = 3 + n_scalars

    if plausible_only:
        assert labels is not None, 'The labels file is necessary to use only the plausible streamlines'

    if labels is not None:
        labels = np.load(labels)
        positive_indices = np.where(labels == 1)[0]
        negative_indices = np.where(labels == 0)[0]

    if idxs is None:
        if plausible_only:
            idxs = positive_indices
        elif non_plausible_only:
            idxs = negative_indices[:len(positive_indices)]
        else:
            idxs = np.arange(nb_streamlines, dtype=np.int32)

    elif isinstance(idxs, int):
        if plausible_only:
            if mode == 'train':
                idxs = np.random.choice(positive_indices, size=idxs, replace=replace)
            else:
                idxs = positive_indices[:idxs]
        elif non_plausible_only:
            if mode == 'train':
                idxs = np.random.choice(negative_indices, size=idxs, replace=replace)
            else:
                idxs = negative_indices[:idxs]
        else:
            if mode == 'train':
                idxs = np.random.choice(np.arange(nb_streamlines), idxs, replace=replace)
            else:
                idxs = np.arange(min(idxs,nb_streamlines))

    elif isinstance(idxs, list):
        idxs = np.array(idxs, dtype=np.int32)

    buffer = np.memmap(trk_fn, dtype='<f4', mode='r', offset=1000)  # after header
    buffer_int32 = buffer.view(np.int32)

    lengths, offsets = parse_lengths_and_offsets(buffer_int32, nb_streamlines, point_size, n_properties)

    # Batch extraction to padded array
    N = len(idxs)
    lens = lengths[idxs]
    if N == 0:
        # edge case
        if return_len:
            return [], np.array([], dtype=np.int32)
        else:
            return []

    max_len = int(lens.max())
    arr = np.zeros((N, max_len, 3), dtype=np.float32)

    # Extract streamlines into arr
    for j, i in enumerate(idxs):
        L = int(lens[j])
        start = int(offsets[i] + 1)
        n_floats = int(lengths[i] * point_size)
        data = buffer[start:start + n_floats]
        s = data.reshape(-1, point_size)[:, :3]
        # s may be smaller than max_len along axis 0
        arr[j, :L] = s

    if apply_affine:
        arr = apply_affine_to_streamlines_batch(arr, rotation, translation)

    # Rebuild list of variable-length streamlines
    streamlines = [arr[j, :int(lens[j])].copy() for j in range(N)]

    if container == 'array':
        streamlines = np.array(streamlines, dtype=object)
    elif container == 'ArraySequence':
        streamlines = nib.streamlines.ArraySequence(streamlines)
    elif container == 'list':
        pass
    elif container == 'array_flat':
        streamlines = np.concatenate(streamlines, axis=0)
        return_len = True
    else:
        raise Exception(f"Unknown container type: {container}")

    if return_len:
        return streamlines, lengths[idxs]
    else:
        return streamlines


def set_random_seed(seed: int) -> None:
    r"""Set random seeds for everything.
    Args:
        seed (int): Random seed.
    """
    print(f"Using random seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_cudnn(deterministic: bool, benchmark: bool) -> None:
    r"""Initialize the cudnn module. The two things to consider is whether to
    use cudnn benchmark and whether to use cudnn deterministic. If cudnn
    benchmark is set, then the cudnn deterministic is automatically false.
    Args:
        deterministic (bool): Whether to use cudnn deterministic.
        benchmark (bool): Whether to use cudnn benchmark.
    """
    cudnn.deterministic = deterministic
    cudnn.benchmark = benchmark
    print('cudnn benchmark: {}'.format(benchmark))
    print('cudnn deterministic: {}'.format(deterministic))


def translate_config(cfg: dict) -> dict:
    for key in cfg:
        if isinstance(cfg[key], dict):
            cfg[key] = translate_config(cfg[key])
        else:
            if cfg[key] == 'n':
                cfg[key] = False
            elif cfg[key] == 'y':
                cfg[key] = True
            elif cfg[key] == 'None':
                cfg[key] = None

    return cfg


def mkdirs(paths, overwrite: bool = False) -> None:
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path, overwrite)
    else:
        mkdir(paths, overwrite)


def mkdir(path, overwrite: bool = False) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if overwrite:
            print(f'Removing old folder {path}... ', end='')
            shutil.rmtree(path)
            print('Done')
            print(f'Creating new folder {path}... ', end='')
            os.makedirs(path)
        else:
            exit('This experiment already exists. Please rename the experiment.'
                 ' The name of the experiment can be given through the flag --name (or -n)')


def directed_hausdorff(A: NDArray[np.floating], B: NDArray[np.floating]) -> float:

    treeB = KDTree(B)
    dist, _ = treeB.query(A, k=1)

    return float(np.max(dist))


def hausdorff_mdf(A: NDArray[np.floating], B: NDArray[np.floating]) -> float:

    direct = directed_hausdorff(A, B)
    flipped = directed_hausdorff(A, B[::-1])

    return min(direct, flipped)


def directed_chamfer(A: NDArray[np.floating], B: NDArray[np.floating], squared: bool=True) -> float:

    tree = KDTree(B)
    distA_to_B, _ = tree.query(A, k=1)

    if squared:
        distA_to_B = distA_to_B ** 2

    return float(np.mean(distA_to_B))


def chamfer_mdf(A: NDArray[np.floating], B: NDArray[np.floating]) -> float:

    direct = directed_chamfer(A, B, squared=False)
    inverse = directed_chamfer(A, B[::-1], squared=False)

    return float(min(direct, inverse))


def reset_all_loggers():
    """
    Close and remove handles of registered loggers.
    Reset the state of the logging system.
    """
    # Get the loggers
    loggers = logging.Logger.manager.loggerDict

    for name, logger in loggers.items():
        if not isinstance(logger, logging.Logger):
            continue

        handlers = logger.handlers[:]
        for h in handlers:
            try:
                h.flush()
            except Exception:
                pass
            try:
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)

        logger.setLevel(logging.NOTSET)
        logger.propagate = False

    # Close the handlers
    root_handlers = logging.getLogger().handlers[:]
    for h in root_handlers:
        try:
            h.flush()
        except Exception:
            pass
        logging.getLogger().removeHandler(h)

    logging.shutdown()

