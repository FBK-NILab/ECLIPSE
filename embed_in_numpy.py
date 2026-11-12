from src.utils import load_model, embed_streamlines, check_embedding_params, check_streamlines


def main(streamlines, checkpoint_path, use_gpu=False, cuda=0, verbose=False, chunk_dim=10000, workers=None):
    """Embed a list of streamlines with ECLIPSE and return the embeddings as a numpy array.

    Args:
        streamlines: list (or dipy ArraySequence) of N streamlines, each an array of shape (n_points, 3).
        checkpoint_path: path of the model checkpoint.
        use_gpu: whether to run the model on the GPU.
        cuda: index of the GPU to use when use_gpu is True.
        verbose: whether to show a progress bar.
        chunk_dim: number of streamlines embedded at once (lower it if you run out of memory).
        workers: number of threads used to embed the chunks in parallel. Used only when > 1 and running on CPU.

    Returns:
        np.ndarray of shape (N, 32), in the same order as the input streamlines.
    """
    check_embedding_params(chunk_dim, workers)
    check_streamlines(streamlines)

    eclipse = load_model(checkpoint_path, use_gpu, cuda, verbose)

    return embed_streamlines(eclipse, streamlines, chunk_dim, verbose, workers=workers)
