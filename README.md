# Edge-Convolution Latent Inference for Brain Tractography Embedding

This repository is the official implementation of "Edge-Convolution Latent Inference for Brain Tractography Embedding", currently under review

## Overview

Streamlines are widely used to represent trajectories and fiber-like structures, but their variable length and geometric complexity make them challenging to model. This work proposes **ECLIPSE**, an autoencoder architecture for streamlines that leverages EdgeConv-style neighborhood aggregation to learn expressive latent representations directly from point-based streamline data.

Key characteristics of the approach:

1. Streamlines are modeled as graphs with point-wise nodes
2. Edge convolution captures local geometric relationships along streamlines
3. A latent embedding enables reconstruction, similarity analysis, and downstream tasks

We also provide an unofficial implementation of our baseline, FINTA[1] for comparison.

## Repo structure
~~~
.  
 ├── src/   
 │ ├── configs/                                         # Configuration files   
 │ ├── dataset/                                         # Dataset implementations  
 │ ├── loops/                                           # Training and test loop implementations  
 │ ├── losses/                                          # Loss implementations  
 │ ├── models/                                          # Model implementations  
 │ ├── splits/                                          # Subject splits of the Bil&Gin dataset  
 │ ├── utils/                                           # Miscellaneous utils functions  
 │ ├── \_\_init__.py  
 ├── Dockerfile                                         # Instructions to build a docker image  
 ├── generate_plots.py                                  # Script for generating plots of latent space and reconstruction error  
 ├── high_loss_visualization.py                         # Script for generating the 3d projection of the top-K worst performing streamlines in reconstruction  
 ├── main.py                                            # Script for training and testing the models  
 ├── plausible_vs_non_plausible_error.py                # Script for testing on plausible AND non-plausible streamlines   
 ├── save_embeddings.py                                 # Script to save model embeddings in trx format  
 ├── tractography_type_analysis.py                      # Script to test performance based on tractography type  
~~~
## Reproducibility
~~~
git clone https://github.com/FBK-NILab/ECLIPSE.git
cd Eclipse
docker build -t \<container_name>:\<tag>
~~~

To run the code, inside docker run:
~~~
python \<script_name> -C \<config_path> -n \<experiment_name>  
~~~

N.B. When testing, the experiment name is used to load the checkpoint. If no experiment name is provided, testing will not work as expected.

### Tested on
The code was executed on a Linux-based system with CUDA 12.x (GPU: Ampere A40)

## Paper
Citation will be provided upon acceptance

## References

[1] Legarreta, J. H., Petit, L., Rheault, F., Theaud, G., Lemaire, C., Descoteaux, M., & Jodoin, P. M. (2021). Filtering in tractography using autoencoders (FINTA). Medical Image Analysis, 72, 102126.
