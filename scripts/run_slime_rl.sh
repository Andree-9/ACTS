#!/bin/bash
# ACTS controller RL training via SLIME.
#
# Usage:
#   bash scripts/run_slime_rl.sh
#
# Edit the Configuration block below to change GPU layout, paths, or hyperparams.

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SLIME_DIR="$PROJECT_DIR/slime"
MODEL_ARGS_ROTARY_BASE=5000000
source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

# =============================================================================
# Configuration
# =============================================================================

# --- GPU ---
# Layout: 8 GPUs split 4 controller + 4 reasoner.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
NUM_GPUS=8
CONTROLLER_GPUS=4
REASONER_GPUS=$((NUM_GPUS - CONTROLLER_GPUS))
TP_SIZE=$(( NUM_GPUS <= 4 ? NUM_GPUS : 4 ))

# --- Models (local paths or Hugging Face repo IDs) ---
CONTROLLER_CKPT="$PROJECT_DIR/checkpoints/sft-controller"
LOAD_CKPT="${LOAD_CKPT:-$CONTROLLER_CKPT}"
REF_CKPT="${REF_CKPT:-$CONTROLLER_CKPT}"
REASONER_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# --- Paths ---
SAVE_PATH="$PROJECT_DIR/checkpoints/rl-controller"
MAX_CHECKPOINTS_TO_KEEP=3
# Full-prompt RL dataset using direct reasoner calibrated budgets.
PROMPT_DATA="$PROJECT_DIR/data/controller_rl_data/controller_rl_data.jsonl"

# --- Reasoner ---
# Keep the reasoner environment deterministic so RL optimizes controller
# behavior rather than reasoner sampling noise.
REASONER_TEMPERATURE=0.0
REASONER_TOP_P=1.0
REASONER_MAX_TOKENS_PER_STEP=512
REASONER_FORCED_CONCLUDE_MAX_TOKENS=256
REASONER_FINAL_ANSWER_MAX_TOKENS=2048

# --- Controller sampling ---
# Keep rollout-time controller sampling aligned with eval.
CONTROLLER_TEMPERATURE=1.0
CONTROLLER_TOP_P=0.9

# --- RL ---
GRADER_BACKEND="math_verify"
MAX_TURNS=500
MAX_REASONING_TRACE_TOKENS=32768
NUM_ROLLOUT=181
ROLLOUT_BATCH_SIZE=32
OVER_SAMPLING_BATCH_SIZE=64
N_SAMPLES_PER_PROMPT=8
GLOBAL_BATCH_SIZE=64
LR=1e-6
# Full-prompt rows are short at rollout start; keep a conservative cap for
# chat-template overhead and future prompt variants.
ROLLOUT_MAX_PROMPT_LEN=10240
TRAIN_MAX_TOKENS_PER_GPU=22528
# Full-continuation RL uses a per-sample response cap:
# max_response_tokens = MAX_ROLLOUT_TOTAL_TOKENS - prompt_tokens.
MAX_ROLLOUT_TOTAL_TOKENS=16384

# --- Reward shaping ---
OVER_BUDGET_GRACE_FRAC=0.10
BUDGET_SAVED_GRACE_FRAC=0.10
CORRECT_OVER_BUDGET_ALPHA=0.5
WRONG_UNDER_BUDGET_ALPHA=0.5

# --- Serving ---
# SLIME HTTP client concurrency multiplier. With 4 controller + 4 reasoner
# engines (1 GPU each), the aggregate client-side capacity scales with each
# side's engine count. This is not a per-worker running request cap.
SGLANG_SERVER_CONCURRENCY=64
# Native SGLang per-worker admission cap. This prevents one worker from
# admitting too many long live requests and exhausting its KV token pool.
SGLANG_MAX_RUNNING_REQUESTS=32

# --- External repos ---
MEGATRON_DIR="$PROJECT_DIR/../Megatron-LM"

# --- Ray ---
RAY_ADDRESS="http://127.0.0.1:8265"
MASTER_ADDR="127.0.0.1"

# =============================================================================
# Validate
# =============================================================================

if [ "$NUM_GPUS" -lt 2 ]; then
    echo "ERROR: NUM_GPUS must be >= 2"; exit 1
fi

if [ "$CONTROLLER_GPUS" -lt 1 ] || [ "$REASONER_GPUS" -lt 1 ]; then
    echo "ERROR: CONTROLLER_GPUS must be in [1, NUM_GPUS-1], got CONTROLLER_GPUS=$CONTROLLER_GPUS NUM_GPUS=$NUM_GPUS"; exit 1
fi

if [[ "$CONTROLLER_CKPT" == /* || "$CONTROLLER_CKPT" == .* ]] && [ ! -d "$CONTROLLER_CKPT" ]; then
    echo "ERROR: Controller checkpoint not found: $CONTROLLER_CKPT"; exit 1
fi

if [[ "$LOAD_CKPT" == /* || "$LOAD_CKPT" == .* ]] && [ ! -d "$LOAD_CKPT" ]; then
    echo "ERROR: Load checkpoint not found: $LOAD_CKPT"; exit 1
fi

if [[ "$REF_CKPT" == /* || "$REF_CKPT" == .* ]] && [ ! -d "$REF_CKPT" ]; then
    echo "ERROR: Reference checkpoint not found: $REF_CKPT"; exit 1
fi

if [[ "$REASONER_MODEL" == /* || "$REASONER_MODEL" == .* ]] && [ ! -d "$REASONER_MODEL" ]; then
    echo "ERROR: Reasoner model not found: $REASONER_MODEL"; exit 1
fi

if [ ! -f "$PROMPT_DATA" ]; then
    echo "ERROR: Prompt data not found: $PROMPT_DATA"; exit 1
fi

mkdir -p "$SAVE_PATH"

# =============================================================================
# Generate configs (derived from above, no need to edit)
# =============================================================================

SGLANG_CONFIG=$(mktemp /tmp/sglang_XXXXXX.yaml)
CUSTOM_CONFIG=$(mktemp /tmp/custom_XXXXXX.yaml)
cleanup() {
    ray stop
    pkill -f sglang
    rm -f "$SGLANG_CONFIG" "$CUSTOM_CONFIG"
}
trap cleanup EXIT INT TERM

cat > "$SGLANG_CONFIG" <<EOF
sglang:
  - name: controller
    update_weights: true
    server_groups:
      - worker_type: regular
        num_gpus: ${CONTROLLER_GPUS}
        num_gpus_per_engine: 1
        overrides:
          mem_fraction_static: 0.90
          attention_backend: triton
          max_running_requests: ${SGLANG_MAX_RUNNING_REQUESTS}
  - name: reasoner
    model_path: ${REASONER_MODEL}
    update_weights: false
    server_groups:
      - worker_type: regular
        num_gpus: ${REASONER_GPUS}
        num_gpus_per_engine: 1
        overrides:
          mem_fraction_static: 0.90
          attention_backend: triton
          max_running_requests: ${SGLANG_MAX_RUNNING_REQUESTS}
EOF

cat > "$CUSTOM_CONFIG" <<EOF
max_turns: ${MAX_TURNS}
max_reasoning_trace_tokens: ${MAX_REASONING_TRACE_TOKENS}
max_rollout_total_tokens: ${MAX_ROLLOUT_TOTAL_TOKENS}
reasoner_model_path: "${REASONER_MODEL}"
reasoner_temperature: ${REASONER_TEMPERATURE}
reasoner_top_p: ${REASONER_TOP_P}
reasoner_max_tokens_per_step: ${REASONER_MAX_TOKENS_PER_STEP}
reasoner_forced_conclude_max_tokens: ${REASONER_FORCED_CONCLUDE_MAX_TOKENS}
reasoner_final_answer_max_tokens: ${REASONER_FINAL_ANSWER_MAX_TOKENS}
grader_backend: "${GRADER_BACKEND}"
grader_timeout_score: 0.0
over_budget_grace_frac: ${OVER_BUDGET_GRACE_FRAC}
budget_saved_grace_frac: ${BUDGET_SAVED_GRACE_FRAC}
correct_over_budget_alpha: ${CORRECT_OVER_BUDGET_ALPHA}
wrong_under_budget_alpha: ${WRONG_UNDER_BUDGET_ALPHA}
EOF

echo "GPU config: ${NUM_GPUS} total, controller=${CONTROLLER_GPUS}, reasoner=${REASONER_GPUS}, TP=${TP_SIZE}"
echo "Controller: ${CONTROLLER_CKPT}"
echo "Load:       ${LOAD_CKPT}"
echo "Reasoner:   ${REASONER_MODEL}"
echo "Save:       ${SAVE_PATH}"

# =============================================================================
# Launch
# =============================================================================

if ! ray job list --address="$RAY_ADDRESS" >/dev/null 2>&1; then
    ray start --head --node-ip-address "$MASTER_ADDR" --num-gpus "$NUM_GPUS" \
        --disable-usage-stats --dashboard-host 0.0.0.0 --dashboard-port 8265
fi

LOG_DIR="$PROJECT_DIR/logs"
SLIME_LOG="${LOG_DIR}/slime_train.log"
mkdir -p "$LOG_DIR"
echo "Logging to: ${SLIME_LOG}"

ray job submit --address="$RAY_ADDRESS" \
    --runtime-env-json="{\"env_vars\":{\"PYTHONPATH\":\"${SLIME_DIR}:${PROJECT_DIR}:${MEGATRON_DIR}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"LIBRARY_PATH\":\"${LIBRARY_PATH:-/usr/lib/x86_64-linux-gnu}\",\"SGLANG_FLASHINFER_WORKSPACE_SIZE\":\"2147483648\",\"SGLANG_ENABLE_JIT_DEEPGEMM\":\"0\",\"SGL_ENABLE_JIT_DEEPGEMM\":\"0\"}}" \
    -- python3 "${SLIME_DIR}/train.py" \
    --actor-num-nodes 1 --actor-num-gpus-per-node "$NUM_GPUS" \
    --rollout-num-gpus "$NUM_GPUS" --colocate \
    "${MODEL_ARGS[@]}" \
    --megatron-to-hf-mode bridge \
    --hf-checkpoint "$CONTROLLER_CKPT" \
    --load "$LOAD_CKPT" \
    --ref-load "$REF_CKPT" \
    --save "$SAVE_PATH" --save-interval 25 --max-checkpoints-to-keep "$MAX_CHECKPOINTS_TO_KEEP" \
    --sglang-config "$SGLANG_CONFIG" \
    --custom-config-path "$CUSTOM_CONFIG" \
    --custom-generate-function-path slime.rollout.acts_rollout.generate_group \
    --custom-rm-path slime.rollout.rm_hub.acts_reward.custom_rm \
    --custom-reward-post-process-path slime.rollout.rm_hub.acts_reward.post_process_rewards \
    --custom-rollout-log-function-path slime.rollout.acts_log.log_rollout_data \
    --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label \
    --apply-chat-template --rollout-shuffle \
    --num-rollout "$NUM_ROLLOUT" --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT" \
    --over-sampling-batch-size "$OVER_SAMPLING_BATCH_SIZE" \
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std \
    --soft-dynamic-sampling-filter \
    --rollout-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN" \
    --rollout-temperature "$CONTROLLER_TEMPERATURE" \
    --rollout-top-p "$CONTROLLER_TOP_P" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --advantage-estimator grpo \
    --disable-grpo-std-normalization \
    --entropy-coef 0.00 --eps-clip 0.2 \
    --optimizer adam --lr "$LR" --lr-decay-style constant \
    --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --tensor-model-parallel-size "$TP_SIZE" --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --use-dynamic-batch-size --max-tokens-per-gpu "$TRAIN_MAX_TOKENS_PER_GPU" \
    --log-probs-chunk-size 16384 \
    --rollout-num-gpus-per-engine 1 \
    --sglang-server-concurrency "$SGLANG_SERVER_CONCURRENCY" \
    --sglang-enable-deterministic-inference \
    --router-policy manual \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
    --attention-backend flash \
    --wandb-always-use-train-step \
    --use-wandb --wandb-project ACTS --wandb-group slime-controller-rl \
    2>&1 | tee "$SLIME_LOG"
