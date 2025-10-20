# Upgraded to a newer base image with Ubuntu 22.04 and a more recent CUDA toolkit
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set the environment to non-interactive to prevent installation prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages for Ubuntu 22.04. The names are the same.
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  python3 \
  python3-pip \
  python3-dev \
  libgl1-mesa-dev \
  cmake \
  libopencv-dev \
  git \
  build-essential \
  ffmpeg && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install all Python packages.
# Pinned numpy to <2.0 to maintain compatibility with the pre-compiled PyTorch wheel.
RUN pip3 install --no-cache-dir \
  torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
  pip3 install --no-cache-dir \
  "numpy<2.0" dlib imutils scipy pandas opencv-python tqdm pretrainedmodels imgaug efficientnet_pytorch vit-pytorch scikit-learn networkx && \
  pip3 install --no-cache-dir -U retinaface_pytorch

# Set the working directory for the application
WORKDIR /app