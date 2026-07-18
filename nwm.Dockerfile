FROM continuumio/miniconda3

# ------------------------------
# System packages
# ------------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    nano \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ------------------------------
# Install mamba
# ------------------------------
RUN conda install -n base -c conda-forge mamba -y

# ------------------------------
# Create NWM environment
# ------------------------------
RUN mamba create -n nwm python=3.10 -y

# Make all following commands run inside nwm env
SHELL ["conda", "run", "-n", "nwm", "/bin/bash", "-c"]

# ------------------------------
# Install PyTorch
# NOTE:
# Your host shows CUDA 12.4, so do not use cu126 for now.
# ------------------------------
RUN pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# ------------------------------
# Install NWM Python dependencies
# ------------------------------
RUN pip install \
    decord \
    einops \
    evo \
    transformers \
    diffusers \
    tqdm \
    timm \
    notebook \
    dreamsim \
    torcheval \
    lpips \
    opencv-python-headless \
    ipywidgets \
    accelerate



RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate nwm" >> /root/.bashrc
# ------------------------------
# Default command
# ------------------------------
CMD ["bash"]