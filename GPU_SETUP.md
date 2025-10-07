# GPU Setup Guide for TUBBA

TUBBA uses PyTorch for its LSTM models, which can be accelerated with GPU support. This guide will help you set up GPU acceleration on different platforms.

## macOS (Apple Silicon)

Apple Silicon Macs (M1/M2/M3) use Metal Performance Shaders (MPS) for GPU acceleration:

```bash
conda install pytorch -c pytorch
```

TUBBA will automatically detect and use MPS when available. No additional setup needed!

## Windows / Linux (NVIDIA GPU)

### Prerequisites

1. **NVIDIA GPU** with CUDA Compute Capability 3.5 or higher
2. **NVIDIA Driver** (latest recommended)
   - Download: https://www.nvidia.com/download/index.aspx
3. **CUDA Toolkit** 11.8 or 12.1 (optional, conda will install cuda libraries)
   - Download: https://developer.nvidia.com/cuda-downloads

### Installation

#### Option 1: Using the install script (Recommended)

```bash
./install.sh
```

When prompted, answer **'y'** to install CUDA support.

#### Option 2: Manual installation

```bash
conda create -n tubba python=3.12
conda activate tubba

# Install core dependencies
conda install -c conda-forge pyqt matplotlib seaborn scikit-learn h5py joblib opencv numpy pandas xgboost tqdm pytables

# Install PyTorch with CUDA 11.8
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# OR for CUDA 12.1:
# conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Verify GPU Setup

After installation, verify CUDA is working:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

Expected output:
```
CUDA available: True
CUDA device: NVIDIA GeForce RTX 3090  # (or your GPU model)
```

### Troubleshooting

**Problem: `torch.cuda.is_available()` returns `False`**

Solutions:
1. **Update NVIDIA drivers**: Download latest from NVIDIA website
2. **Verify CUDA installation**: Run `nvidia-smi` in terminal
   - Should show GPU info and CUDA version
   - If command not found, reinstall NVIDIA drivers
3. **Reinstall PyTorch with CUDA**:
   ```bash
   conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia --force-reinstall
   ```
4. **Check CUDA compatibility**:
   - Run `nvidia-smi` to see your CUDA version
   - Install matching PyTorch version (11.8 or 12.1)

**Problem: Out of memory errors**

- Close other GPU-intensive applications
- Reduce batch size in training (if you modify training parameters)
- Monitor GPU usage: `nvidia-smi -l 1` (updates every 1 second)

## CPU-Only Installation

If you don't have a GPU or prefer CPU-only:

```bash
conda install pytorch cpuonly -c pytorch
```

TUBBA will work fine on CPU, but training will be slower.

## Performance Notes

- **GPU Training**: ~5-10x faster than CPU for LSTM training
- **macOS (MPS)**: ~3-5x speedup on Apple Silicon
- **Inference**: GPU acceleration provides modest improvements (~2x)

The main benefit of GPU is during model training. For annotation-only workflows, CPU is sufficient.