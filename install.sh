#!/bin/bash
# TUBBA Installation Script

echo "🎯 Installing TUBBA..."
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment 'tubba'..."
conda create -n tubba python=3.12 -y

# Activate environment
echo "🔧 Activating environment..."
eval "$(conda shell.bash hook)"
conda activate tubba

# Install packages
echo "📥 Installing dependencies..."
conda install -c conda-forge pyqt matplotlib seaborn scikit-learn h5py joblib opencv numpy pandas xgboost tqdm pytables -y

# Install PyTorch
echo "🔥 Installing PyTorch..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - use MPS (Metal Performance Shaders) for Apple Silicon
    echo "   Installing PyTorch for macOS (with MPS support for Apple Silicon)..."
    conda install pytorch -c pytorch -y
else
    # Linux/Windows - ask user about GPU
    echo ""
    echo "Do you have an NVIDIA GPU and want to use CUDA acceleration? (y/n)"
    read -r use_gpu

    if [[ "$use_gpu" == "y" || "$use_gpu" == "Y" ]]; then
        echo "   Installing PyTorch with CUDA support..."
        echo "   Note: Make sure you have NVIDIA drivers and CUDA 11.8+ installed!"
        conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        echo "   Installing PyTorch (CPU-only version)..."
        conda install pytorch cpuonly -c pytorch -y
    fi
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "To use TUBBA:"
echo "  1. conda activate tubba"
echo "  2. cd src"
echo "  3. python TUBBA.py"
echo ""

if [[ "$OSTYPE" != "darwin"* ]] && [[ "$use_gpu" == "y" || "$use_gpu" == "Y" ]]; then
    echo "📌 GPU Setup Notes:"
    echo "   - Verify CUDA is working: python -c 'import torch; print(torch.cuda.is_available())'"
    echo "   - If False, ensure NVIDIA drivers and CUDA toolkit are properly installed"
    echo "   - CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
    echo ""
fi