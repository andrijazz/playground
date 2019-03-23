# SETUP GPU
###########################################################################
# OS Ubuntu 16.04
# GPU Tesla K80
# CUDA 9.0
# Tensorflow version 1.12.0
###########################################################################
# check if graphics card is connected to PCI bus
sudo lspci | grep -i nvidia

# status should be DIS...
sudo lshw -numeric -C display

# install appropriate nvidia driver for your gpu based on operating system and CUDA version
# https://www.nvidia.com/Download/index.aspx?lang=en-us
wget http://us.download.nvidia.com/tesla/384.183/nvidia-diag-driver-local-repo-ubuntu1604-384.183_1.0-1_amd64.deb
sudo dpkg -i nvidia-diag-driver-local-repo-ubuntu1604-384.183_1.0-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-drivers
sudo reboot

# tf things https://www.tensorflow.org/install/gpu

# Add NVIDIA package repository
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo apt install ./cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt update

# Install CUDA and tools. Include optional NCCL 2.x
sudo apt install cuda9.0 cuda-cublas-9-0 cuda-cufft-9-0 cuda-curand-9-0 \
    cuda-cusolver-9-0 cuda-cusparse-9-0 libcudnn7=7.2.1.38-1+cuda9.0 \
    libnccl2=2.2.13-1+cuda9.0 cuda-command-line-tools-9-0

# Optional: Install the TensorRT runtime (must be after CUDA install)
# sudo apt update
# sudo apt install libnvinfer4=4.1.2-1+cuda9.0

# if everything is ok you should be able to run
sudo nvidia-smi
