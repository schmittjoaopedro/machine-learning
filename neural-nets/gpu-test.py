import tensorflow as tf

# Install NVIDIA Driver
# https://forums.developer.nvidia.com/t/nvidia-geforce-1050ti-is-not-detected-after-update-ubuntu-20-04/155331/4
# https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/

# Installing CUDA Drivers
# https://medium.com/analytics-vidhya/installing-any-version-of-cuda-on-ubuntu-and-using-tensorflow-and-torch-on-gpu-b1a954500786
# https://forums.developer.nvidia.com/t/installing-new-nvidia-drivers-and-cuda-and-cudnn-on-an-nvidia-geforce-1050-ti/278023
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
# Latest CUDA Toolkit 12.4
# https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local

# Show GPU details: nvidia-smi
# Show CUDA details: nvcc --version
# Make sure both versions match

# Installing TensorFlow libraries for GPU
# https://www.tensorflow.org/install/gpu

# To run this script make sure to export the following environment variables:
# export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=${CUDNN_PATH}/lib

# Install latest version of XLS
# https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu

for device in tf.config.list_physical_devices():
    print(device)
