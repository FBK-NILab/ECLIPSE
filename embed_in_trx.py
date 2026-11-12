import os
import sys

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram

from src.utils import InferenceOnlyParameters, load_model, embed_streamlines, check_embedding_params


def embed_tractogram(eclipse, sft, chunk_dim=10000, verbose=False, workers=None):
    embeddings = embed_streamlines(eclipse, sft.streamlines, chunk_dim, verbose, workers=workers)
    return StatefulTractogram.from_sft(sft.streamlines, sft, data_per_streamline={'embeddings': embeddings})


def get_output_file(data_path, out_path):
    stem = os.path.splitext(os.path.basename(data_path))[0]
    return os.path.join(out_path, stem + '_with_embedding.trx')


def main(checkpoint_path, data_path, out_path, reference='same', save_out=False, use_gpu=False, cuda=0, verbose=False,
         chunk_dim=10000, workers=None):

    if data_path.lower().endswith('.tck') and reference == 'same':
        raise ValueError('A .tck tractogram has no spatial header: a reference image must be provided')
    check_embedding_params(chunk_dim, workers)

    eclipse = load_model(checkpoint_path, use_gpu, cuda, verbose)
    sft = load_tractogram(data_path, reference)
    new_sft = embed_tractogram(eclipse, sft, chunk_dim, verbose, workers=workers)

    if save_out:
        os.makedirs(out_path, exist_ok=True)
        save_tractogram(new_sft, get_output_file(data_path, out_path))

    return new_sft


if __name__ == '__main__':

    params = InferenceOnlyParameters()
    args = params.parse()

    if not os.path.exists(args.data_path):
        params.parser.error(f'The specified data path {args.data_path} does not exist')
    if not os.path.exists(args.checkpoint_path):
        params.parser.error(f'The specified checkpoint path {args.checkpoint_path} does not exist')
    if args.save and args.out_path is None:
        params.parser.error('When --save-out is set, the output path must be specified with --out-path')
    if args.data_path.lower().endswith('.tck') and args.reference == 'same':
        params.parser.error('A .tck tractogram requires a reference image, please specify it with --reference')
    if args.chunk_dim < 1:
        params.parser.error(f'--chunk-dim must be a positive integer, got {args.chunk_dim}')
    if args.workers is not None and args.workers < 1:
        params.parser.error(f'--workers must be a positive integer, got {args.workers}')

    checkpoint_path = args.checkpoint_path
    data_path = args.data_path
    save_out = args.save
    out_path = args.out_path
    reference = args.reference
    verbose = args.verbose
    cuda = args.cuda
    use_gpu = cuda is not None
    chunk_dim = args.chunk_dim
    workers = args.workers

    if use_gpu:
        if workers is not None:
            print('When using the GPU, the model will automatically work with only 1 GPU. The number of workers is ignored')
        workers = None

    if save_out:
        out_file = get_output_file(data_path, out_path)
        if os.path.exists(out_file):
            ans = 'n'
            if not args.auto_yes:
                ans = input(f'The output file {out_file} already exists. Do you want to overwrite it? (y/n) ').lower()
                if ans == 'y':
                    print('Consider using the flag -y in the command to automatically overwrite old results when specifying the same output path')
            if args.auto_yes or ans == 'y':
                os.remove(out_file)
            else:
                print('Please change the output path and re-run the script')
                sys.exit(1)

    result = main(checkpoint_path, data_path, out_path, reference, save_out, use_gpu, cuda, verbose, chunk_dim, workers=workers)
