#!/bin/bash
# Set up the conda environment for controller SFT training with OpenRLHF.
#
# This is a SEPARATE conda env from SLIME (see env_setup_slime.sh), but it is
# pinned to the SAME Python (3.12), CUDA (12.8), PyTorch (2.9.1) and the SAME
# prebuilt Flash-Attention wheel (2.8.3) so the two envs stay binary-compatible.
# We install plain OpenRLHF (no vLLM) since the controller is trained with the
# DeepSpeed-based `openrlhf.cli.train_sft` entrypoint (see run_openrlhf_sft.sh).
#
# Prerequisites: miniconda/anaconda installed, CUDA drivers available.
#
# Usage:
#   ./scripts/env_setup_openrlhf.sh

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OPENRLHF_DIR="$PROJECT_DIR/openrlhf"

# =============================================================================
# 1. Create conda environment (Python 3.12, matching the SLIME env)
# =============================================================================

conda create -n openrlhf python=3.12 pip -y
eval "$(conda shell.bash hook)"
conda activate openrlhf

# =============================================================================
# 2. CUDA toolkit via conda (12.8 — same as env_setup_slime.sh; provides nvcc
#    for DeepSpeed JIT-compiled fused ops)
# =============================================================================

conda install -c nvidia/label/cuda-12.8.0 -c conda-forge cuda cuda-nvtx cuda-nvtx-dev nccl cudnn -y
export CUDA_HOME="$CONDA_PREFIX"

# =============================================================================
# 3. PyTorch (same version + CUDA build as env_setup_slime.sh)
# =============================================================================

pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# =============================================================================
# 4. Flash-Attention (same prebuilt wheel as env_setup_slime.sh — no source build).
#    OpenRLHF pins flash-attn==2.8.3; installing the prebuilt 2.8.3 wheel first
#    satisfies that requirement, so `pip install -e .` below will NOT rebuild it.
# =============================================================================

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# =============================================================================
# 5. OpenRLHF (vendored, editable). Pulls its remaining pinned deps
#    (deepspeed==0.18.5, transformers==4.57.6, ray, ...) from requirements.txt.
#
#    OpenRLHF's requirements leave `torch` unpinned and pin `flash-attn==2.8.3`,
#    both of which are already satisfied by the builds installed above, so pip
#    leaves them in place. We additionally pass a constraints file pinning the
#    exact torch build so the resolver can never silently pull a CPU/PyPI torch
#    and clobber our CUDA 12.8 wheels.
# =============================================================================

CONSTRAINTS="$(mktemp)"
cat > "$CONSTRAINTS" <<EOF
torch==2.9.1
torchvision==0.24.1
torchaudio==2.9.1
EOF

cd "$OPENRLHF_DIR"
PIP_CONSTRAINT="$CONSTRAINTS" pip install -e .
rm -f "$CONSTRAINTS"

# Sanity check: confirm our CUDA builds survived the OpenRLHF install.
python - <<'PY'
import torch
print("torch:", torch.__version__, "| CUDA build:", torch.version.cuda)
assert torch.__version__.startswith("2.9.1"), "torch was changed by the install!"
assert (torch.version.cuda or "").startswith("12"), "non-CUDA-12 torch build!"
import flash_attn
print("flash_attn:", flash_attn.__version__)
assert flash_attn.__version__.startswith("2.8.3"), "flash-attn was changed!"
print("OK: torch 2.9.1 (cu12x) + flash-attn 2.8.3 intact after OpenRLHF install.")
PY

# =============================================================================
# 6. Training extras
# =============================================================================

pip install math_verify wandb

# Log in to Weights and Biases (training scripts log to wandb by default)
wandb login

echo "============================================================"
echo "OpenRLHF environment setup complete."
echo "Activate with: conda activate openrlhf"
echo "============================================================"
