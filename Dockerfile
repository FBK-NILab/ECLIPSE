FROM nvcr.io/nvidia/pytorch:25.08-py3

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
 && DEBIAN_FRONTEND=noninteractive apt-get -qqy install python3-pip wget git less nano libsm6 libxext6 libxrender-dev software-properties-common tmux htop
RUN DEBIAN_FRONTEND=noninteractive add-apt-repository ppa:quentiumyt/nvtop && DEBIAN_FRONTEND=noninteractive apt-get -qq install nvtop

# Create a non-root user
ARG username=nilab
ARG uid=1014
ARG gid=1014
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER

RUN groupadd -g $GID nilab

RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --gid $GID \
    --home $HOME \
    $USER

RUN chown -R ${USER} /home/${USER}

USER $USER

RUN pip install tqdm 
RUN pip install nibabel 
RUN pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126 
RUN pip install torch_geometric 
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html 
RUN pip install PyYAML
RUN pip install pprintpp
RUN pip install numba
RUN pip install tensorboard
RUN pip install torchinfo
RUN pip install tabulate
RUN pip install onnx onnxscript
RUN pip install matplotlib
RUN pip install dipy
RUN pip install dtaidistance

WORKDIR $HOME

RUN mkdir -p $HOME/src $HOME/data/
