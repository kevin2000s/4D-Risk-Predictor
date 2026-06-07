#!/bin/bash
set -e

echo "============================================"
echo "  4D Risk Predictor - Build Executable"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Please install Python 3.9+ first."
    exit 1
fi

echo "[1/5] Python detected:"
python3 --version
echo ""

# Create virtual environment
echo "[2/5] Creating build environment..."
rm -rf venv_build
python3 -m venv venv_build

# Activate and install deps
echo "[3/5] Installing dependencies..."
source venv_build/bin/activate
pip install -q pyinstaller pandas numpy scikit-learn scipy matplotlib joblib ttkbootstrap Pillow

# Build
echo "[4/5] Building executable with PyInstaller..."
python -m PyInstaller 4D_Risk_Predictor.spec

# Cleanup
echo "[5/5] Cleaning up..."
rm -rf build venv_build

echo ""
echo "============================================"
echo "  Build Complete!"
echo "============================================"
echo ""
echo "Output: dist/4D_Risk_Predictor/"
echo ""
