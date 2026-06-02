#!/bin/bash
# Download training datasets from HuggingFace.
#
# Usage:
#   ./scripts/get_data.sh              # download all datasets
#   ./scripts/get_data.sh sft          # download SFT data only
#   ./scripts/get_data.sh rl           # download RL data only
#   ./scripts/get_data.sh all          # download everything

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")/data"

SFT_REPO="yuuxia/controller-sft-data"
RL_REPO="yuuxia/controller-rl-data"
SFT_DIR="$DATA_DIR/controller_sft_data"
RL_DIR="$DATA_DIR/controller_rl_data"

download_dataset() {
    local repo="$1" dest="$2" name="$3"
    echo "Downloading ${name} from ${repo} → ${dest}"
    mkdir -p "$dest"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${repo}',
    repo_type='dataset',
    local_dir='${dest}',
)
print('Done: ${name}')
"
}

TARGET="${1:-all}"

case "$TARGET" in
    sft)
        download_dataset "$SFT_REPO" "$SFT_DIR" "controller SFT data"
        ;;
    rl)
        download_dataset "$RL_REPO" "$RL_DIR" "controller RL data"
        ;;
    all)
        download_dataset "$SFT_REPO" "$SFT_DIR" "controller SFT data"
        download_dataset "$RL_REPO" "$RL_DIR" "controller RL data"
        ;;
    *)
        echo "Usage: $0 [sft|rl|all]"
        exit 1
        ;;
esac

echo "All downloads complete."
