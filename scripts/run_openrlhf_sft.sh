#!/bin/bash
# Controller SFT Training Script
#
# Trains the reasoning control agent on multi-turn SFT data using OpenRLHF's train_sft module.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/openrlhf:$PROJECT_DIR:$PYTHONPATH"

deepspeed --module openrlhf.cli.train_sft \
    --max_len 10240 \
    --dataset ./data/controller_sft_data/controller_sft_data.jsonl \
    --dataset_num_proc 32 \
    --input_key messages \
    --apply_chat_template \
    --multiturn \
    --train_batch_size 64 \
    --micro_train_batch_size 1 \
    --pretrain Qwen/Qwen3-4B-Instruct-2507 \
    --save_path "$PROJECT_DIR/checkpoints/sft-controller" \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 1 \
    --max_epochs 1 \
    --param_dtype bf16 \
    --learning_rate 1e-5 \
    --packing_samples \
    --attn_implementation flash_attention_2 \
    --use_wandb True \
    --wandb_project ACTS \
    --wandb_run_name controller_sft
