#!/bin/bash
# setup_env.sh

# Create conda environment with native ARM64 Python
conda create -n deepfake-detection python=3.11 -y

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepfake-detection

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install other dependencies
pip install opencv-python mediapipe scipy pandas numpy matplotlib scikit-learn tqdm jupyter kagglehub pillow

# Verify MPS availability
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}'); print(f'MPS Built: {torch.backends.mps.is_built()}')"