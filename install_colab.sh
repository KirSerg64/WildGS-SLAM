#!/bin/bash
# ============================================================
# WildGS-SLAM — Google Colab Installation Script
# ============================================================
# Usage (inside a Colab notebook cell):
#   !git clone --recursive https://github.com/GradientSpaces/WildGS-SLAM.git
#   %cd WildGS-SLAM
#   !bash install_colab.sh
# ============================================================
# NOTE: This script targets CUDA 11.8 + PyTorch 2.1.0, which
# is the tested configuration.  If your Colab runtime has a
# different CUDA driver you may need to adjust the cu* tags
# in the pip URLs below.
# ============================================================

set -e  # abort on first error

# ── 0. Sanity-check: must be run from the repo root ─────────
if [ ! -f "setup.py" ] || [ ! -d "thirdparty" ]; then
    echo "ERROR: Please run this script from the WildGS-SLAM repository root."
    exit 1
fi

# ── 1. Show CUDA / driver info ───────────────────────────────
echo "=== CUDA version ==="
nvcc --version 2>/dev/null || nvidia-smi 2>/dev/null || echo "CUDA tools not found on PATH"

# ── 2. Core numeric library (must precede torch install) ─────
echo "=== Installing numpy <2.0 ==="
pip install -q "numpy==1.26.3"

# ── 3. PyTorch 2.1.0 (cu118 build) ──────────────────────────
echo "=== Installing PyTorch 2.1.0 (cu118) ==="
pip install -q \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# ── 4. torch-scatter & xformers ─────────────────────────────
echo "=== Installing torch-scatter ==="
pip install -q torch-scatter \
    -f https://pytorch-geometric.com/whl/torch-2.1.0+cu118.html

echo "=== Installing xformers ==="
pip install -q -U \
    "xformers==0.0.22.post7+cu118" \
    --index-url https://download.pytorch.org/whl/cu118

# ── 5. Setuptools (pinned to avoid editable-install issues) ──
echo "=== Installing setuptools 78.1.1 ==="
pip install -q "setuptools==78.1.1"

# ── 6. Thirdparty CUDA extensions ────────────────────────────
echo "=== Building lietorch ==="
pip install -q -e thirdparty/lietorch/

echo "=== Building diff-gaussian-rasterization-w-pose ==="
pip install -q -e thirdparty/diff-gaussian-rasterization-w-pose/

echo "=== Building simple-knn ==="
pip install -q -e thirdparty/simple-knn/

# ── 7. Verify core imports ───────────────────────────────────
echo "=== Verifying core imports ==="
python - <<'EOF'
import torch
import lietorch
import simple_knn
import diff_gaussian_rasterization
print("torch version       :", torch.__version__)
print("CUDA available      :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device         :", torch.cuda.get_device_name(0))
print("All core imports OK.")
EOF

# ── 8. Droid backends & Python requirements ──────────────────
echo "=== Installing droid backends ==="
pip install -q -e .

echo "=== Installing Python requirements ==="
pip install -q -r requirements.txt

# ── 9. MMCV (metric depth estimator) ─────────────────────────
echo "=== Installing mmcv-full (cu118 / torch2.1.0) ==="
pip install -q mmcv-full \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

# ── 10. Pretrained model reminder ────────────────────────────
echo ""
echo "============================================================"
echo " Installation complete!"
echo ""
echo " NEXT STEP — download the pretrained DROID model:"
echo "   mkdir -p pretrained"
echo "   # Download droid.pth from Google Drive:"
echo "   # https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh"
echo "   # and place it at:  pretrained/droid.pth"
echo "============================================================"
