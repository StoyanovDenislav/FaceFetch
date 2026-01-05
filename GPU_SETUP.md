# GPU Acceleration Setup Guide

You have an **NVIDIA RTX 3060** which can significantly speed up facial recognition!

## Prerequisites

### 1. Install CUDA Toolkit (Required)

**For Arch Linux:**

```bash
sudo pacman -S cuda cudnn cmake
```

**For Ubuntu/Debian:**

```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA toolkit
sudo apt-get install cuda-toolkit-12-3 libcudnn8 libcudnn8-dev
```

### 2. Verify CUDA Installation

```bash
nvcc --version
nvidia-smi
```

## GPU Setup for Facial Recognition

### Option 1: Automatic Setup (Recommended)

```bash
cd /home/denkata/Desktop/pythonCV
./setup_gpu.sh
```

### Option 2: Manual Installation

1. **Activate virtual environment:**

```bash
source /home/denkata/Desktop/pythonCV/venv/bin/activate
```

2. **Install build tools:**

```bash
pip install cmake
```

3. **Uninstall current dlib:**

```bash
pip uninstall -y dlib
```

4. **Build dlib with CUDA support:**

**Method A - Using pip (simpler):**

```bash
pip install dlib --no-binary :all: --verbose
```

**Method B - From source (more control):**

```bash
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build && cd build
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
cmake --build . --config Release --parallel $(nproc)
cd ..
python setup.py install
```

## Verify GPU is Working

Run this command:

```bash
python -c "import dlib; print('GPU Available:', dlib.DLIB_USE_CUDA); print('GPU Devices:', dlib.cuda.get_num_devices() if dlib.DLIB_USE_CUDA else 0)"
```

Expected output:

```
GPU Available: True
GPU Devices: 1
```

## Performance Comparison

**CPU (Current):**

- Face detection: ~30-50ms per frame
- Face encoding: ~100-200ms per face

**GPU (Expected with RTX 3060):**

- Face detection: ~10-15ms per frame (3x faster)
- Face encoding: ~30-50ms per face (4x faster)

## Important Notes

1. **First run after GPU setup will be slower** - CUDA needs to compile kernels
2. **Memory usage increases** - GPU needs ~2-4GB VRAM
3. **Model parameter**: face_recognition uses 'hog' or 'cnn' models
   - HOG: CPU-optimized, faster on CPU
   - CNN: GPU-optimized, much faster on GPU

## Update Code to Use CNN Model (GPU-optimized)

After GPU is enabled, modify face detection calls:

**Change from:**

```python
face_locations = face_recognition.face_locations(frame, model="hog")
```

**Change to:**

```python
face_locations = face_recognition.face_locations(frame, model="cnn")
```

This will use the CNN model which is optimized for GPU acceleration.

## Troubleshooting

### CUDA not found during build

```bash
export CUDA_HOME=/opt/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### CMake can't find CUDA

```bash
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/opt/cuda -DDLIB_USE_CUDA=1
```

### GPU not being used even after installation

Check if GPU is accessible:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## After GPU Setup

Restart the web server:

```bash
export QT_QPA_PLATFORM=xcb
python /home/denkata/Desktop/pythonCV/web_server.py
```

You should see: "🚀 GPU Acceleration ENABLED"
