FROM nvcr.io/nvidia/pytorch:21.06-py3

# Vars that should be set with -e flag when running container:
# WANDB_API_KEY
# NCCL_P2P_DISABLE=1
# NCCL_SOCKET_IFNAME=eno2

# install basic dependencies
RUN apt-get update && apt-get install -y nano git && apt-get clean
RUN apt-get install -y libsm6 libxext6 libxrender-dev

# install other dependencies
RUN pip install ray=="1.2.0" \
wandb=="0.10.27" \
adabelief-pytorch=="0.1.0" \
pytorch-fid=="0.1.1" \
opencv-python=="4.2.0.34" \
schedule=="0.6.0" \
mysql-connector=="2.2.9" \
h5py=="3.2.1" \
easydict=="1.9" \
selenium=="3.141.0" \
yapf=="0.30.0" \
azure-core=="1.15.0" \
azure-storage=="0.36.0" \
imutils=="0.5.4"
RUN pip install --upgrade --force-reinstall azure-storage-blob=="12.8.1"

# set env
ENV SRC_ROOT=/opt/trendage
ENV PROJECT=trendage
ENV STORAGE=/tmp
ENV PROJECT_STORAGE_ROOT=$STORAGE/$PROJECT
ENV DATASETS=$PROJECT_STORAGE_ROOT/datasets
ENV MODELS=$PROJECT_STORAGE_ROOT/models
ENV LOG=$PROJECT_STORAGE_ROOT/log
ENV RAY_LOG=$LOG/ray
ENV PYTHONPATH=$SRC_ROOT:$PYTHONPATH

## creating dirs
RUN mkdir -p $SRC_ROOT && mkdir -p $DATASETS && mkdir -p $MODELS && mkdir -p $RAY_LOG

# set workdir
WORKDIR $SRC_ROOT
