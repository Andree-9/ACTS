#!/bin/bash
# ACTS single-run inference demo.
#
# Launches a controller + reasoner SGLang server pair and runs ACTS on one
# (reasoner, benchmark, budget) setting — for illustration / quick testing,
# not the full sweep (use run_acts_eval.sh for that).
#
# Usage:
#   ./scripts/run_acts_inference.sh \
#       --controller yuuxia/acts-controller \
#       --reasoner   deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#       --benchmark  aime2024 \
#       --budget     10000
#
# Any unset flag falls back to the defaults below; the full benchmark is run.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EVAL_SCRIPT="${PROJECT_DIR}/evaluation/run_acts_eval.py"

# --- Defaults (overridable via flags) ---
CONTROLLER_MODEL="yuuxia/acts-controller"
REASONER_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
BENCHMARK="aime2024"
BUDGET=10000
SEED=1234
CONTROLLER_GPUS="0"
REASONER_GPUS="1"
OUTPUT_DIR="${PROJECT_DIR}/results/acts_inference"

# --- Servers / generation (match run_acts_eval.sh) ---
CONTROLLER_PORT=31000
REASONER_PORT=31001
MEM_FRACTION=0.9
MAX_TURNS=500
MAX_REASONING_TRACE_TOKENS=32768
CONTROLLER_TEMPERATURE=0.7
CONTROLLER_TOP_P=0.8
REASONER_TEMPERATURE=0.6
REASONER_TOP_P=0.95
REASONER_MAX_TOKENS_PER_STEP=512
FORCED_CONCLUDE_MAX_TOKENS=256
FINAL_ANSWER_MAX_TOKENS=2048
SCORE_BACKEND="math_verify"
TIMEOUT_SCORE="0.0"
CONCURRENCY=32

# --- Parse flags ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --controller)      CONTROLLER_MODEL="$2"; shift 2 ;;
        --reasoner)        REASONER_MODEL="$2";   shift 2 ;;
        --benchmark)       BENCHMARK="$2";        shift 2 ;;
        --budget)          BUDGET="$2";           shift 2 ;;
        --seed)            SEED="$2";             shift 2 ;;
        --controller-gpus) CONTROLLER_GPUS="$2";  shift 2 ;;
        --reasoner-gpus)   REASONER_GPUS="$2";    shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";       shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "${PROJECT_DIR}/logs" "$OUTPUT_DIR"

# --- Server management ---
CONTROLLER_PID=""
REASONER_PID=""
cleanup() {
    echo ""
    echo "Cleaning up servers..."
    [ -n "$CONTROLLER_PID" ] && kill "$CONTROLLER_PID" 2>/dev/null && echo "Killed controller (PID $CONTROLLER_PID)"
    [ -n "$REASONER_PID" ] && kill "$REASONER_PID" 2>/dev/null && echo "Killed reasoner (PID $REASONER_PID)"
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_server() {
    local url="$1" name="$2" max_wait=900 elapsed=0
    echo "Waiting for $name server at $url ..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -sf "${url}/health_generate" > /dev/null 2>&1; then
            echo "$name server ready (${elapsed}s)"; return 0
        fi
        sleep 2; elapsed=$((elapsed + 2))
    done
    echo "ERROR: $name server did not start within ${max_wait}s"; return 1
}

launch_server() {
    local gpus="$1" model="$2" port="$3" logname="$4"
    echo "Launching $logname server on port $port ($model) ..."
    CUDA_VISIBLE_DEVICES=$gpus python3 -m sglang.launch_server \
        --model-path "$model" \
        --host 0.0.0.0 --port "$port" \
        --tp 1 --dp 1 \
        --mem-fraction-static "$MEM_FRACTION" \
        --attention-backend triton \
        --trust-remote-code \
        > "${PROJECT_DIR}/logs/${logname}_server.log" 2>&1 &
}

echo "============================================================"
echo "ACTS inference demo"
echo "  controller : $CONTROLLER_MODEL"
echo "  reasoner   : $REASONER_MODEL"
echo "  benchmark  : $BENCHMARK   budget: $BUDGET"
echo "  output     : $OUTPUT_DIR"
echo "============================================================"

launch_server "$REASONER_GPUS" "$REASONER_MODEL" "$REASONER_PORT" "reasoner"
REASONER_PID=$!
launch_server "$CONTROLLER_GPUS" "$CONTROLLER_MODEL" "$CONTROLLER_PORT" "controller"
CONTROLLER_PID=$!

wait_for_server "http://localhost:$REASONER_PORT" "Reasoner"
wait_for_server "http://localhost:$CONTROLLER_PORT" "Controller"

python3 "$EVAL_SCRIPT" \
    --controller_url "http://localhost:$CONTROLLER_PORT" \
    --reasoner_url "http://localhost:$REASONER_PORT" \
    --controller_dp 1 \
    --reasoner_dp 1 \
    --controller_model "$CONTROLLER_MODEL" \
    --reasoner_model "$REASONER_MODEL" \
    --benchmark "$BENCHMARK" \
    --budget "$BUDGET" \
    --max_turns "$MAX_TURNS" \
    --max_reasoning_trace_tokens "$MAX_REASONING_TRACE_TOKENS" \
    --controller_temperature "$CONTROLLER_TEMPERATURE" \
    --controller_top_p "$CONTROLLER_TOP_P" \
    --reasoner_temperature "$REASONER_TEMPERATURE" \
    --reasoner_top_p "$REASONER_TOP_P" \
    --reasoner_max_tokens_per_step "$REASONER_MAX_TOKENS_PER_STEP" \
    --forced_conclude_max_tokens "$FORCED_CONCLUDE_MAX_TOKENS" \
    --final_answer_max_tokens "$FINAL_ANSWER_MAX_TOKENS" \
    --score_backend "$SCORE_BACKEND" \
    --timeout_score "$TIMEOUT_SCORE" \
    --seed "$SEED" \
    --concurrency "$CONCURRENCY" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Done. Results written under: $OUTPUT_DIR"
