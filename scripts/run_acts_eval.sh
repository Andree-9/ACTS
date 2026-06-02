#!/bin/bash
# ACTS Evaluation — SGLang HTTP Server Version
#
# Launches two SGLang HTTP servers (controller + reasoner) and runs the
# async evaluation script that talks to both via HTTP. All samples are
# processed concurrently for high throughput.
#
# GPU allocation:
#   Controller server: first half of visible GPUs, one DP worker per GPU
#   Reasoner server:   second half of visible GPUs, one DP worker per GPU
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
AGGREGATE_SCRIPT="${PROJECT_DIR}/evaluation/aggregate_repeated_results.py"

# =============================================================================
# Configuration (edit these directly)
# =============================================================================

# --- Models ---
# Controller is shared across all reasoners. Each reasoner is launched
# on-demand when the sweep moves to its first triple for that model.
CONTROLLER_MODEL="${PROJECT_DIR}/checkpoints/rl-controller"
CONTROLLER_RESULTS_SUBDIR="results_acts"

declare -A REASONER_MODEL_PATHS=(
    ["ds_15b"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    ["ds_7b"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    ["qwen3_8b"]="Qwen/Qwen3-8B"
)
declare -A RESULTS_ROOTS=(
    ["ds_15b"]="results_deepseek_1.5B"
    ["ds_7b"]="results_deepseek_7b"
    ["qwen3_8b"]="results_qwen3_8B"
)

# --- GPU (8 GPUs: 4 controller engines + 4 reasoner engines, 1 GPU each) ---
CONTROLLER_GPUS="0,1,2,3"
REASONER_GPUS="4,5,6,7"
CONTROLLER_TP=1
REASONER_TP=1
CONTROLLER_DP=4
REASONER_DP=4

# --- Server ---
CONTROLLER_PORT=30004
REASONER_PORT=30005
MEM_FRACTION=0.9

# --- Benchmark / Budget sweep ---
# Baseline budgets = direct-generation avg think tokens per benchmark, measured
# per reasoner via run_vanilla_eval.sh. Keyed as "<model_key>:<benchmark>" so
# the same table serves every model in REASONER_MODEL_PATHS.
declare -A BASELINE_BUDGETS=(
    # DeepSeek-R1-Distill-Qwen-1.5B
    ["ds_15b:aime2024"]=16182
    ["ds_15b:amc"]=10374
    ["ds_15b:math500"]=5036
    ["ds_15b:olympiadbench"]=11966
    ["ds_15b:gpqa_diamond"]=8887
    # DeepSeek-R1-Distill-Qwen-7B
    ["ds_7b:aime2024"]=12497
    ["ds_7b:amc"]=7362
    ["ds_7b:math500"]=3950
    ["ds_7b:olympiadbench"]=8025
    ["ds_7b:gpqa_diamond"]=7950
    # Qwen3-8B
    ["qwen3_8b:aime2024"]=13755
    ["qwen3_8b:amc"]=9726
    ["qwen3_8b:math500"]=4836
    ["qwen3_8b:olympiadbench"]=10519
    ["qwen3_8b:gpqa_diamond"]=8790
)

# Budget ratios (percent of the per-benchmark baseline) swept for each
# (reasoner, benchmark) pair. Grouped by model to minimize reasoner server
# restarts; the main loop relaunches the reasoner only when model_key changes.
MODEL_ORDER=(ds_15b ds_7b qwen3_8b)
BENCHMARK_ORDER=(math500 amc olympiadbench gpqa_diamond aime2024)
SWEEP="20 40 60 80 100 120 140 160"
declare -A MODEL_BENCH_RATIO_SETS=(
    ["ds_15b:math500"]="$SWEEP"
    ["ds_15b:amc"]="$SWEEP"
    ["ds_15b:olympiadbench"]="$SWEEP"
    ["ds_15b:gpqa_diamond"]="$SWEEP"
    ["ds_15b:aime2024"]="$SWEEP"

    ["ds_7b:math500"]="$SWEEP"
    ["ds_7b:amc"]="$SWEEP"
    ["ds_7b:olympiadbench"]="$SWEEP"
    ["ds_7b:gpqa_diamond"]="$SWEEP"
    ["ds_7b:aime2024"]="$SWEEP"

    ["qwen3_8b:math500"]="$SWEEP"
    ["qwen3_8b:amc"]="$SWEEP"
    ["qwen3_8b:olympiadbench"]="$SWEEP"
    ["qwen3_8b:gpqa_diamond"]="$SWEEP"
    ["qwen3_8b:aime2024"]="$SWEEP"
)

# Resolve to (model_key, benchmark, budget_tokens) work list.
WORK_LIST=()
for _mkey in "${MODEL_ORDER[@]}"; do
    if [ -z "${REASONER_MODEL_PATHS[$_mkey]:-}" ]; then
        echo "ERROR: unknown model key '$_mkey'" >&2
        exit 1
    fi
    for _bench in "${BENCHMARK_ORDER[@]}"; do
        _base="${BASELINE_BUDGETS[${_mkey}:${_bench}]:-}"
        _ratios="${MODEL_BENCH_RATIO_SETS[${_mkey}:${_bench}]:-}"
        if [ -z "$_base" ]; then
            echo "ERROR: no baseline for ${_mkey}:${_bench}" >&2
            exit 1
        fi
        if [ -z "$_ratios" ]; then
            echo "SKIP: no ratio set for ${_mkey}:${_bench} (commented out)"
            continue
        fi
        for _ratio in $_ratios; do
            _budget=$(( (_base * _ratio) / 100 ))
            WORK_LIST+=("${_mkey}:${_bench}:${_budget}")
        done
    done
done

LIMIT=0  # 0 = all samples

# --- Generation ---
MAX_TURNS=500
MAX_REASONING_TRACE_TOKENS=32768
CONTROLLER_TEMPERATURE=0.7
CONTROLLER_TOP_P=0.8
REASONER_TEMPERATURE=0.6
REASONER_TOP_P=0.95
REASONER_MAX_TOKENS_PER_STEP=512
FORCED_CONCLUDE_MAX_TOKENS=256
FINAL_ANSWER_MAX_TOKENS=2048

# --- Scoring ---
SCORE_BACKEND="math_verify"
TIMEOUT_SCORE="0.0"
BASE_SEED=1234

# --- Concurrency ---
CONCURRENCY=64

# --- Output ---
# OUTPUT_DIR is resolved per-triple in the main loop from RESULTS_ROOTS[model_key].

# =============================================================================
# Server management
# =============================================================================

CONTROLLER_PID=""
REASONER_PID=""

cleanup() {
    echo ""
    echo "Cleaning up servers..."
    [ -n "$CONTROLLER_PID" ] && kill "$CONTROLLER_PID" 2>/dev/null && echo "Killed controller (PID $CONTROLLER_PID)"
    [ -n "$REASONER_PID" ] && kill "$REASONER_PID" 2>/dev/null && echo "Killed reasoner (PID $REASONER_PID)"
    wait 2>/dev/null || true
}
trap cleanup EXIT

launch_reasoner() {
    local model_path="$1"
    echo ""
    echo "Launching reasoner server on port $REASONER_PORT ($model_path) ..."
    CUDA_VISIBLE_DEVICES=$REASONER_GPUS python3 -m sglang.launch_server \
        --model-path "$model_path" \
        --host 0.0.0.0 \
        --port "$REASONER_PORT" \
        --tp "$REASONER_TP" \
        --dp "$REASONER_DP" \
        --mem-fraction-static "$MEM_FRACTION" \
        --attention-backend triton \
        --trust-remote-code \
        > "${PROJECT_DIR}/logs/reasoner_server.log" 2>&1 &
    REASONER_PID=$!
    echo "Reasoner PID: $REASONER_PID"
    wait_for_server "http://localhost:$REASONER_PORT" "Reasoner"
}

kill_reasoner() {
    if [ -n "$REASONER_PID" ]; then
        kill "$REASONER_PID" 2>/dev/null && echo "Killed reasoner (PID $REASONER_PID)"
        # `wait` returns 128+SIGNUM (e.g. 143 for SIGTERM) for a signal-killed
        # child; swallow that so `set -e` doesn't abort the sweep here.
        wait "$REASONER_PID" 2>/dev/null || true
        REASONER_PID=""
        # Give SGLang a moment to release GPU memory before the next launch.
        sleep 5
    fi
}

get_repeat_count() {
    case "$1" in
        aime2024|amc) echo 5 ;;
        gpqa_diamond) echo 3 ;;
        *) echo 1 ;;
    esac
}

wait_for_server() {
    local url="$1"
    local name="$2"
    local max_wait=900  # 15 minutes
    local elapsed=0

    echo "Waiting for $name server at $url ..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -sf "${url}/health_generate" > /dev/null 2>&1; then
            echo "$name server ready (${elapsed}s)"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo "ERROR: $name server did not start within ${max_wait}s"
    return 1
}

# =============================================================================
# Launch controller server (shared across all reasoner models)
# =============================================================================

echo "============================================================"
echo "ACTS Evaluation (Async SGLang)"
echo "============================================================"
echo "Controller: $CONTROLLER_MODEL"
echo "Controller GPUs: $CONTROLLER_GPUS (TP=$CONTROLLER_TP, DP=$CONTROLLER_DP)"
echo "Reasoner GPUs:   $REASONER_GPUS (TP=$REASONER_TP, DP=$REASONER_DP)"
echo "Work list (${#WORK_LIST[@]} runs):"
for _w in "${WORK_LIST[@]}"; do echo "  - $_w"; done
echo "Concurrency: $CONCURRENCY"
echo "============================================================"

mkdir -p "${PROJECT_DIR}/logs"

echo ""
echo "Launching controller server on port $CONTROLLER_PORT ..."
CUDA_VISIBLE_DEVICES=$CONTROLLER_GPUS python3 -m sglang.launch_server \
    --model-path "$CONTROLLER_MODEL" \
    --host 0.0.0.0 \
    --port "$CONTROLLER_PORT" \
    --tp "$CONTROLLER_TP" \
    --dp "$CONTROLLER_DP" \
    --mem-fraction-static "$MEM_FRACTION" \
    --attention-backend triton \
    --trust-remote-code \
    > "${PROJECT_DIR}/logs/controller_server.log" 2>&1 &
CONTROLLER_PID=$!
echo "Controller PID: $CONTROLLER_PID"

wait_for_server "http://localhost:$CONTROLLER_PORT" "Controller"
echo ""
echo "Controller ready. Reasoner will be launched per model."
echo ""

# =============================================================================
# Run evaluation for each (model_key, benchmark, budget) triple
# =============================================================================

EVAL_SCRIPT="${PROJECT_DIR}/evaluation/run_acts_eval.py"

# Hardcoded list of (model_key, benchmark, budget) triples already completed.
# Edit this list to resume from where we left off without re-running finished work.
COMPLETED_TRIPLES=()

CURRENT_MODEL_KEY=""
REASONER_MODEL=""

for work in "${WORK_LIST[@]}"; do

MODEL_KEY="${work%%:*}"
_rest="${work#*:}"
BENCHMARK="${_rest%%:*}"
BUDGET="${_rest##*:}"
REPEAT_COUNT=$(get_repeat_count "$BENCHMARK")

OUTPUT_DIR="${PROJECT_DIR}/${RESULTS_ROOTS[$MODEL_KEY]}/${CONTROLLER_RESULTS_SUBDIR}"
RUN_DIR="${OUTPUT_DIR}/controlled_${BENCHMARK}_budget${BUDGET}"

_already_done=0
for _done in "${COMPLETED_TRIPLES[@]}"; do
    if [ "$_done" = "${MODEL_KEY}:${BENCHMARK}:${BUDGET}" ]; then
        _already_done=1
        break
    fi
done
if [ "$_already_done" -eq 1 ]; then
    echo ""
    echo "SKIP: ${MODEL_KEY}:${BENCHMARK} budget $BUDGET (marked completed)"
    continue
fi

# Relaunch reasoner when model_key changes.
if [ "$MODEL_KEY" != "$CURRENT_MODEL_KEY" ]; then
    kill_reasoner
    REASONER_MODEL="${REASONER_MODEL_PATHS[$MODEL_KEY]}"
    launch_reasoner "$REASONER_MODEL"
    CURRENT_MODEL_KEY="$MODEL_KEY"
fi

mkdir -p "$RUN_DIR"
REPEAT_JSONLS=()
REPEAT_ELAPSED_SECONDS=()
REPEAT_SEEDS=()

echo ""
echo "========================================================================"
echo "Starting: ${MODEL_KEY} | $BENCHMARK with budget $BUDGET"
echo "Reasoner:   $REASONER_MODEL"
echo "Output dir: $OUTPUT_DIR"
echo "========================================================================"
echo "Repeats: $REPEAT_COUNT"

EXTRA_ARGS=()
if [ "$LIMIT" -gt 0 ]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi

for (( repeat_idx=1; repeat_idx<=REPEAT_COUNT; repeat_idx++ )); do
    RUN_SEED=$(( BASE_SEED + (repeat_idx - 1) * 1000 ))
    REPEAT_SEEDS+=("$RUN_SEED")
    REPEAT_START_TS=$(date +%s)
    SUFFIX="_repeat${repeat_idx}"

    echo ""
    echo "Repeat $repeat_idx/$REPEAT_COUNT (seed $RUN_SEED)"

    python3 "$EVAL_SCRIPT" \
        --controller_url "http://localhost:$CONTROLLER_PORT" \
        --reasoner_url "http://localhost:$REASONER_PORT" \
        --controller_dp "$CONTROLLER_DP" \
        --reasoner_dp "$REASONER_DP" \
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
        --seed "$RUN_SEED" \
        --concurrency "$CONCURRENCY" \
        --output_dir "$OUTPUT_DIR" \
        --output_suffix "$SUFFIX" \
        "${EXTRA_ARGS[@]}"

    REPEAT_END_TS=$(date +%s)
    REPEAT_ELAPSED_SECONDS+=($(( REPEAT_END_TS - REPEAT_START_TS )))
    REPEAT_JSONLS+=("${RUN_DIR}/merged_results${SUFFIX}.jsonl")
done

if [ "$REPEAT_COUNT" -gt 1 ]; then
    python3 "$AGGREGATE_SCRIPT" \
        --inputs "${REPEAT_JSONLS[@]}" \
        --output_jsonl "${RUN_DIR}/merged_results.jsonl" \
        --summary_path "${RUN_DIR}/results.txt" \
        --mode controlled \
        --benchmark "$BENCHMARK" \
        --budget "$BUDGET" \
        --repeat_seeds "${REPEAT_SEEDS[@]}" \
        --repeat_elapsed_seconds "${REPEAT_ELAPSED_SECONDS[@]}"
else
    cp "${REPEAT_JSONLS[0]}" "${RUN_DIR}/merged_results.jsonl"
    cp "${RUN_DIR}/results_repeat1.txt" "${RUN_DIR}/results.txt"
fi

echo "Done with ${MODEL_KEY} | $BENCHMARK (budget $BUDGET)!"

done

echo ""
echo "========================================================================"
echo "All (model, benchmark, budget) triples completed!"
echo "========================================================================"
