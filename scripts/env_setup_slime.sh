#!/bin/bash
# Set up the conda environment for SLIME RL training.
#
# Prerequisites: miniconda/anaconda installed, CUDA drivers available.
#
# Usage:
#   ./scripts/env_setup_slime.sh                    # default: clone repos to ~/
#   BASE_DIR=/data/libs ./scripts/env_setup_slime.sh  # custom clone directory

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SLIME_DIR="$PROJECT_DIR/slime"
BASE_DIR="${BASE_DIR:-$HOME}"

SGLANG_COMMIT="bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2"
MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"

# =============================================================================
# 1. Create conda environment
# =============================================================================

conda create -n slime python=3.12 pip -y
eval "$(conda shell.bash hook)"
conda activate slime

# =============================================================================
# 2. CUDA toolkit via conda
# =============================================================================

conda install -c nvidia/label/cuda-12.8.0 -c conda-forge cuda cuda-nvtx cuda-nvtx-dev nccl cudnn -y
export CUDA_HOME="$CONDA_PREFIX"

# =============================================================================
# 3. PyTorch
# =============================================================================

pip install cuda-python==13.1.0
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# =============================================================================
# 4. SGLang
# =============================================================================

cd "$BASE_DIR"
if [ ! -d sglang ]; then
    git clone https://github.com/sgl-project/sglang.git
fi
cd sglang && git checkout "$SGLANG_COMMIT"
pip install -e "python[all]"

# =============================================================================
# 5. CUDA extensions (slow — compiles from source)
# =============================================================================

pip install cmake ninja

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"

NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

# =============================================================================
# 6. Other dependencies
# =============================================================================

pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install flash-linear-attention==0.4.1
pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall
pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation

# =============================================================================
# 7. Megatron-LM
# =============================================================================

cd "$BASE_DIR"
if [ ! -d Megatron-LM ]; then
    git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
fi
cd Megatron-LM && git checkout "$MEGATRON_COMMIT"
pip install -e .

# =============================================================================
# 8. SLIME
# =============================================================================

cd "$SLIME_DIR"
pip install -e .

# =============================================================================
# 9. Compatibility fixes
# =============================================================================

pip install "numpy<2"

# =============================================================================
# 10. Patches
# =============================================================================

cd "$BASE_DIR/sglang"
git apply "$SLIME_DIR/docker/patch/v0.5.9/sglang.patch"

cd "$BASE_DIR/Megatron-LM"
git apply "$SLIME_DIR/docker/patch/v0.5.9/megatron.patch"

# =============================================================================
# 11. Training extras
# =============================================================================

pip install math_verify wandb

# Log in to Weights and Biases (training scripts log to wandb by default)
wandb login

echo "============================================================"
echo "Environment setup complete."
echo "Activate with: conda activate slime"
echo "============================================================"
