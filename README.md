# ECLIPSE: inference only

This branch has the inference-only code of ECLIPSE. It embeds each streamline of a tractogram into a 32-dimensional vector using a pretrained checkpoint. It is meant to be lightweight and easy to include in other codebases.

There are two entry points:

| Script | Input | Output | Use it when |
|---|---|---|---|
| `embed_in_trx.py` | A tractogram file (`.trx`, `.trk`, `.tck`) | A `.trx` tractogram with the embeddings stored per streamline | You work with tractogram files |
| `embed_in_numpy.py` | A list of streamlines already in memory | A NumPy array of shape `(N, 32)` | You already have the streamlines loaded in your own pipeline |

> **Note:** both scripts embed **a single tractogram** per call. To process several files, or data organized in a different way (e.g. one folder per subject), import the `main` functions into your own script and call them in a loop. See [Processing multiple tractograms](#processing-multiple-tractograms).

## Installation

The code was developed with Python 3.10.

```bash
pip install -r requirements.txt
```

`requirements.txt` installs PyTorch from PyPI. For a GPU build matching your CUDA version, install PyTorch first following the [official instructions](https://pytorch.org/get-started/locally/), e.g. for CUDA 12.6:


## Checkpoint

Download the pretrained ECLIPSE checkpoint and pass its path to the scripts. Only the encoder weights are used to compute the embeddings. You can download the `ECLIPSE.pth.tar` checkpoint from [zenodo](https://doi.org/10.5281/zenodo.17981408).

## Usage

### `embed_in_trx.py`: from a tractogram file

```bash
python embed_in_trx.py \
    --checkpoint-path ECLIPSE.pth.tar \
    --data-path sub01.trx \
    --save-out --out-path results
```

This writes `results/sub01_with_embedding.trx`. It has the same streamlines as the input and stores the embeddings in `data_per_streamline['embeddings']`, an array of shape `(N, 32)`.

| Argument | Required | Default | Description |
|---|---|---|---|
| `--checkpoint-path` | yes | | Path of the model checkpoint |
| `--data-path` | yes | | Tractogram to embed (`.trx`, `.trk` or `.tck`) |
| `--save-out` | no | off | Save the output tractogram in `.trx` format |
| `--out-path` | with `--save-out` | | Output folder, created if it does not exist |
| `--reference` | for `.tck` | `same` | Reference image (e.g. the T1) used to load `.tck` files. Ignored for `.trk` and `.trx` |
| `--cuda` | no | CPU | Index of the GPU to use, e.g. `--cuda 0` |
| `--chunk-dim` | no | `10000` | Number of streamlines embedded at once. Lower it if you run out of memory |
| `--workers` | no | `None` | Number of threads used to embed the chunks in parallel. Used only on CPU and when greater than 1 |
| `-y` | no | off | Overwrite the output file without asking if it already exists |
| `-v` | no | off | Verbose output with a progress bar |

Example with a `.tck` file on the GPU:

```bash
python embed_in_trx.py \
    --checkpoint-path ECLIPSE.pth.tar \
    --data-path sub01.tck --reference sub01_T1.nii.gz \
    --save-out --out-path results --cuda 0 -v
```

The output file name always appends in `_with_embedding`, before the extension whatever the input format. This means the input file can never be overwritten. The output format is always `.trx`.

#### From Python

```python
from embed_in_trx import main

sft = main(
    checkpoint_path='ECLIPSE.pth.tar',
    data_path='sub01.trx',
    out_path='results',      # used only when save_out=True
    save_out=True,
)
embeddings = sft.data_per_streamline['embeddings']   # (N, 32)
```

`main` has these arguments: `checkpoint_path`, `data_path`, `out_path`, `reference='same'`, `save_out=False`, `use_gpu=False`, `cuda=0`, `verbose=False`, `chunk_dim=10000` and `workers=None`. It returns a dipy `StatefulTractogram`.

### `embed_in_numpy.py`: from streamlines in memory

```python
from embed_in_numpy import main

# streamlines: list of N arrays, each of shape (n_points, 3); the number of points can differ between streamlines
embeddings = main(streamlines, checkpoint_path='ECLIPSE.pth.tar')   # np.ndarray, shape (N, 32)
```

`main` has these arguments: `streamlines`, `checkpoint_path`, `use_gpu=False`, `cuda=0`, `verbose=False`, `chunk_dim=10000` and `workers=None`. The rows of the output are in the same order as the input streamlines. A dipy `ArraySequence` (e.g. `sft.streamlines`) is also accepted.

The streamline coordinates should be in the same space used by `embed_in_trx.py`. dipy's `load_tractogram` loads streamlines in RASMM (world coordinates, in millimeters) by default.

## Processing multiple tractograms

Both scripts handle one tractogram per call on purpose. Your files may be organized in many different ways, so neither script assumes a particular folder layout. To embed many files, call the `main` functions from your own script. For example, with one folder per subject:

```python
import glob
import os

from embed_in_trx import main

for path in sorted(glob.glob('dataset/*/tractogram.trx')):
    subject = os.path.basename(os.path.dirname(path))
    main(
        checkpoint_path='ECLIPSE.pth.tar',
        data_path=path,
        out_path=os.path.join('results', subject),
        save_out=True,
        use_gpu=True, cuda=0,
    )
```

Each call to `main` reloads the checkpoint. If you embed many tractograms, you can load the model once and call the lower-level functions in `src.utils` directly:

```python
from src.utils import load_model, embed_streamlines

eclipse = load_model('ECLIPSE.pth.tar', use_gpu=True, cuda=0, verbose=False)
for streamlines in my_streamline_sets:
    embeddings = embed_streamlines(eclipse, streamlines, chunk_dim=10000)   # (N, 32)
```

## Notes on performance

- **GPU:** pass `--cuda <id>` (or `use_gpu=True, cuda=<id>`). `--workers` is ignored on the GPU.
- **CPU:** `--workers N` embeds `N` chunks at the same time using threads. Whether this is faster than the default depends on your machine and data, so compare both on a representative tractogram.
- **Memory:** if you run out of memory, lower `--chunk-dim`.
- **Reproducibility:** results do not depend on how the streamlines are split into chunks, because the model runs in evaluation mode. 
