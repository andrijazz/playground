#!/usr/bin/env bash

###############################################################################
# OS Ubuntu 18.04
# 1) setup nvidia drivers
# 2) guide how to create and mount data disk https://docs.microsoft.com/en-us/azure/virtual-machines/linux/attach-disk-portal#connect-to-the-linux-vm-to-mount-the-new-disk
# 3) low resolution problem https://www.youtube.com/watch?v=RfwVWU84cnQ
###############################################################################


# -----------------------------------------------------------------------------
# Install tools
# -----------------------------------------------------------------------------
mkdir tools && cd tools
sudo apt-get update
sudo apt-get install git build-essential screen unzip zip python3-pip tmux

pip3 install --user pipenv
sudo cp ~/.local/bin/pipenv /usr/bin/

## install azcopy
wget https://aka.ms/downloadazcopy-v10-linux
tar -xvf downloadazcopy-v10-linux
sudo cp "${PWD}/azcopy_linux_amd64_10.4.0/azcopy" "/usr/bin"
sudo chown $USER:$USER /usr/bin/azcopy


# -----------------------------------------------------------------------------
# Install nvidia drivers and cuda
# -----------------------------------------------------------------------------

# check if graphics card is connected to PCI bus
sudo lspci | grep -i nvidia

# status should be DIS...
sudo lshw -numeric -C display

# install appropriate nvidia driver for your gpu based on operating system
# https://www.nvidia.com/Download/index.aspx?lang=en-us
# wget http://us.download.nvidia.com/tesla/418.116.00/NVIDIA-Linux-x86_64-418.116.00.run
# chmod +x NVIDIA-Linux-x86_64-418.116.00.run
# sudo ./NVIDIA-Linux-x86_64-418.116.00.run
# sudo reboot

# if everything is ok you should be able to run
sudo nvidia-smi

# tf things ... https://www.tensorflow.org/install/gpu
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

# IMPORTANT: in official tf docs installing nvidia-driver is the next step. However, we already installed it in a first
# step so we should skip this step. Therefore, I'm commenting it out
# Install NVIDIA driver
sudo apt-get install --no-install-recommends nvidia-driver-418
# Reboot. Check that GPUs are visible using the command: nvidia-smi
sudo reboot

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    cuda-10-1 \
    libcudnn7=7.6.4.38-1+cuda10.1  \
    libcudnn7-dev=7.6.4.38-1+cuda10.1


# Install TensorRT. Requires that libcudnn7 is installed above.
sudo apt-get install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 \
    libnvinfer-dev=6.0.1-1+cuda10.1 \
    libnvinfer-plugin6=6.0.1-1+cuda10.1

# Reboot. Check that GPUs are visible using the command: nvidia-smi
sudo reboot

# check if everything is ok again ...
sudo nvidia-smi


# ------------------------------------------------------------------------------------
# Install Docker
# ------------------------------------------------------------------------------------
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
sudo apt update
sudo apt install docker-ce
sudo systemctl status docker

# run docker without sudo
sudo usermod -aG docker ${USER}
su - ${USER}

# ------------------------------------------------------------------------------------
# Install Nvidia Docker
# ------------------------------------------------------------------------------------
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
