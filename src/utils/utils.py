import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from ..models import get_model


def get_edge_index(n_points):
    idx = torch.arange(n_points - 1, dtype=torch.long)
    top = torch.cat([idx, idx + 1])
    bot = torch.cat([idx + 1, idx])

    return torch.stack([top, bot], dim=0).contiguous()


def get_geometric_data(streamlines):

    lines = [torch.as_tensor(line, dtype=torch.float32) for line in streamlines]

    data_list = []

    for line in lines:
        n_points = line.shape[0]
        eidx = get_edge_index(n_points)

        d = Data()
        d.x = line
        d.pos = line
        d.edge_index = eidx
        data_list.append(d)

    return data_list


def check_embedding_params(chunk_dim, workers):
    if chunk_dim < 1:
        raise ValueError(f'chunk_dim must be a positive integer, got {chunk_dim}')
    if workers is not None and workers < 1:
        raise ValueError(f'workers must be a positive integer, got {workers}')


def check_streamlines(streamlines):
    for i, line in enumerate(streamlines):
        shape = np.shape(line)
        if len(shape) != 2 or shape[1] != 3:
            raise ValueError(f'Each streamline must have shape (n_points, 3), streamline {i} has shape {shape}')


def load_model(checkpoint_path, use_gpu, cuda, verbose):
    model_config = {
        'input_size': 3,
        'hidden_dims_ec': [64, 64, 64],
        'hidden_dims_dec': [128],
        'embedding_size': 32,
        'aggr': 'max',
        'batch_norm': True,
        'activation': 'relu'
    }
    eclipse = get_model(model_config)

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    eclipse.load_state_dict(state_dict['model'], strict=True)

    eclipse.eval()
    if use_gpu:
        eclipse.to(f'cuda:{cuda}')

    if verbose:
        print('Model state dict loaded')

    return eclipse


@torch.no_grad()  # grad mode is thread-local, so it must be set here and not only in the caller
def model_embed(eclipse, data_chunk, device):
    batch = Batch.from_data_list(data_chunk).to(device)
    return eclipse.embed(batch).cpu().numpy()


@torch.no_grad()
def embed_streamlines(eclipse, streamlines, chunk_dim=10000, verbose=False, workers=None):
    """Embed a list of streamlines (each an (N_points, 3) array) with ECLIPSE.

    Runs on the model's device. Chunks are embedded in parallel threads only when workers > 1 and the model is on CPU.
    Returns an np.ndarray of shape (n_streamlines, embedding_size), in the same order as the input.
    """
    n_streamlines = len(streamlines)
    if n_streamlines == 0:
        return np.empty((0, eclipse.embedding_size), dtype=np.float32)

    device = next(eclipse.parameters()).device
    geometric_data = get_geometric_data(streamlines)
    chunks = math.ceil(n_streamlines / chunk_dim)
    data_chunks = [geometric_data[i * chunk_dim:(i + 1) * chunk_dim] for i in range(chunks)]
    desc = f'Embedding data chunks (each chunk is {chunk_dim} streamlines)'

    parallel = workers is not None and workers > 1 and device.type == 'cpu'
    if not parallel:
        iterator = tqdm(data_chunks, desc=desc, dynamic_ncols=True) if verbose else data_chunks
        embeddings = [model_embed(eclipse, chunk, device) for chunk in iterator]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(lambda chunk: model_embed(eclipse, chunk, device), data_chunks)  # keeps the chunk order
            if verbose:
                results = tqdm(results, total=chunks, desc=desc, dynamic_ncols=True)
            embeddings = list(results)

    return np.concatenate(embeddings, axis=0)
