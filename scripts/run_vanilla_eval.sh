#!/bin/bash
# Direct Model Evaluation - Parallel Data Split Version (SGLang)
#
# Runs a reasoning model directly on benchmark problems (no controller).
# Uses SGLang offline Engine (consistent with slime eval).
# Uses the model's own chat template via tokenizer.apply_chat_template().
# Splits data across multiple GPUs for parallel evaluation.
#
set -e

# =============================================================================
# Configuration (edit these directly)
# =============================================================================

# Model
MODEL="Qwen/Qwen3-8B"

# Benchmarks to evaluate
BENCHMARKS=("amc" "aime2024" "math500" "olympiadbench" "gpqa_diamond")
# Benchmarks whose prior runs are accepted as-is and will be skipped this pass.
COMPLETED_BENCHMARKS=()
LIMIT=0  # 0 = all samples

# Parallelism
NUM_WORKERS=8
GPUS_PER_WORKER=1

# Generation
MAX_TOKENS=0  # 0 = use dataset default
TEMPERATURE=0.6
TOP_P=0.95
BATCH_SIZE=16
MEM_FRACTION=0.9

# Scoring
SCORE_BACKEND="math_verify"
TIMEOUT_SCORE="0.0"
BASE_SEED=1234

# =============================================================================
# Run (no need to edit below)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EVAL_SCRIPT="${PROJECT_DIR}/evaluation/run_vanilla_eval.py"
AGGREGATE_SCRIPT="${PROJECT_DIR}/evaluation/aggregate_repeated_results.py"

# Derive a safe directory name from the model (e.g. Qwen/Qwen3-8B -> Qwen3-8B)
MODEL_SHORT="${MODEL##*/}"
OUTPUT_DIR="${PROJECT_DIR}/results_qwen3_8B/results_direct_generation"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Direct Model Evaluation (SGLang)"
echo "============================================================"
echo "Model:        $MODEL"
echo "Benchmarks:   ${BENCHMARKS[*]}"
echo "Workers:      $NUM_WORKERS  (GPUs/worker: $GPUS_PER_WORKER)"
echo "Temperature:  $TEMPERATURE  top_p: $TOP_P"
echo "Batch size:   $BATCH_SIZE"
echo "Mem fraction: $MEM_FRACTION"
echo "Score:        $SCORE_BACKEND"
echo "Output:       $OUTPUT_DIR"
echo "============================================================"

OVERALL_START_TS=$(date +%s)

get_repeat_count() {
    case "$1" in
        aime2024|amc) echo 5 ;;
        gpqa_diamond) echo 3 ;;
        *) echo 1 ;;
    esac
}

get_visible_gpu_ids() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        python3 - <<'PY'
import os
raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
items = [part.strip() for part in raw.split(",") if part.strip()]
for item in items:
    print(item)
PY
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index --format=csv,noheader
        return
    fi
}

mapfile -t VISIBLE_GPU_IDS < <(get_visible_gpu_ids)
TOTAL_VISIBLE_GPUS=${#VISIBLE_GPU_IDS[@]}

if [ "$TOTAL_VISIBLE_GPUS" -eq 0 ]; then
    echo "Could not detect any visible GPUs."
    exit 1
fi

TOTAL_GPUS_NEEDED=$(( NUM_WORKERS * GPUS_PER_WORKER ))
if [ "$TOTAL_GPUS_NEEDED" -gt "$TOTAL_VISIBLE_GPUS" ]; then
    echo "Requested $TOTAL_GPUS_NEEDED GPU slots, but only $TOTAL_VISIBLE_GPUS visible GPUs are available."
    exit 1
fi

for BENCHMARK in "${BENCHMARKS[@]}"; do

_skip=0
for _done in "${COMPLETED_BENCHMARKS[@]}"; do
    if [ "$_done" = "$BENCHMARK" ]; then
        _skip=1
        break
    fi
done
if [ "$_skip" -eq 1 ]; then
    echo ""
    echo "SKIP: $BENCHMARK (marked completed in COMPLETED_BENCHMARKS)"
    continue
fi

echo ""
echo "###################################################################"
echo "# Benchmark: $BENCHMARK"
echo "###################################################################"
echo ""

BENCHMARK_START_TS=$(date +%s)

# Count total samples
echo "Counting samples in $BENCHMARK..."
TOTAL_SAMPLES=$(python3 -c "
import sys; sys.path.insert(0, '$PROJECT_DIR')
from evaluation import DATASET_CONFIGS
from datasets import load_dataset
cfg = DATASET_CONFIGS['$BENCHMARK']
ds = load_dataset(cfg['path'], split=cfg['split'])
print(len(ds))
" 2>/dev/null)

if [ "$LIMIT" -gt 0 ] && [ "$LIMIT" -lt "$TOTAL_SAMPLES" ]; then
    TOTAL_SAMPLES=$LIMIT
fi

echo "Total samples: $TOTAL_SAMPLES"
echo "Number of workers: $NUM_WORKERS"

SAMPLES_PER_WORKER=$(( (TOTAL_SAMPLES + NUM_WORKERS - 1) / NUM_WORKERS ))
echo "Samples per worker: ~$SAMPLES_PER_WORKER"

RUN_DIR="${OUTPUT_DIR}/direct_${BENCHMARK}_${MODEL_SHORT}"
mkdir -p "$RUN_DIR"
echo "Output directory: $RUN_DIR"
echo ""

# Get dataset max_tokens if not overridden
BENCH_MAX_TOKENS=$MAX_TOKENS
if [ "$BENCH_MAX_TOKENS" -eq 0 ]; then
    BENCH_MAX_TOKENS=$(python3 -c "
import sys; sys.path.insert(0, '$PROJECT_DIR')
from evaluation import DATASET_CONFIGS
print(DATASET_CONFIGS['$BENCHMARK']['max_gen_toks'])
" 2>/dev/null)
    echo "Using dataset max_tokens: $BENCH_MAX_TOKENS"
fi

REPEAT_COUNT=$(get_repeat_count "$BENCHMARK")
echo "Repeats: $REPEAT_COUNT"

REPEAT_SEEDS=()

JOB_REPEAT=()
JOB_WORKER=()
JOB_START=()
JOB_LIMIT=()
JOB_SEED=()

for (( repeat_idx=1; repeat_idx<=REPEAT_COUNT; repeat_idx++ )); do
    RUN_SEED=$(( BASE_SEED + (repeat_idx - 1) * 1000 ))
    REPEAT_SEEDS+=("$RUN_SEED")
    for (( i=0; i<NUM_WORKERS; i++ )); do
        START=$(( i * SAMPLES_PER_WORKER ))

        if [ $i -eq $((NUM_WORKERS - 1)) ]; then
            WORKER_LIMIT=$(( TOTAL_SAMPLES - START ))
        else
            WORKER_LIMIT=$SAMPLES_PER_WORKER
        fi

        if [ "$WORKER_LIMIT" -le 0 ]; then
            continue
        fi

        WORKER_SEED=$(( RUN_SEED + i ))
        JOB_REPEAT+=("$repeat_idx")
        JOB_WORKER+=("$i")
        JOB_START+=("$START")
        JOB_LIMIT+=("$WORKER_LIMIT")
        JOB_SEED+=("$WORKER_SEED")
    done
done

TOTAL_JOBS=${#JOB_REPEAT[@]}
echo "Queued jobs: $TOTAL_JOBS"
echo "Repeat worker files will be saved under: $RUN_DIR"

declare -a SLOT_PID SLOT_REPEAT SLOT_WORKER SLOT_START SLOT_LIMIT SLOT_SEED SLOT_LOG SLOT_GPU
ACTIVE_JOBS=0
NEXT_JOB_IDX=0
FAILED=0

launch_job() {
    local slot_idx="$1"
    local job_idx="$2"
    local repeat_idx="${JOB_REPEAT[$job_idx]}"
    local worker_idx="${JOB_WORKER[$job_idx]}"
    local start_idx="${JOB_START[$job_idx]}"
    local worker_limit="${JOB_LIMIT[$job_idx]}"
    local worker_seed="${JOB_SEED[$job_idx]}"

    local gpu_start=$(( slot_idx * GPUS_PER_WORKER ))
    local worker_gpu_ids=("${VISIBLE_GPU_IDS[@]:$gpu_start:$GPUS_PER_WORKER}")
    local worker_gpus
    worker_gpus=$(IFS=,; echo "${worker_gpu_ids[*]}")

    local output_file="${RUN_DIR}/worker_repeat${repeat_idx}_${worker_idx}.jsonl"
    local log_file="${RUN_DIR}/worker_repeat${repeat_idx}_${worker_idx}.log"

    echo "Launch slot $slot_idx: repeat $repeat_idx/$REPEAT_COUNT, worker $worker_idx, samples [$start_idx, $((start_idx + worker_limit))), GPUs: $worker_gpus, seed: $worker_seed"

    python3 "$EVAL_SCRIPT" \
        --model "$MODEL" \
        --benchmark "$BENCHMARK" \
        --start "$start_idx" \
        --limit "$worker_limit" \
        --max_tokens "$BENCH_MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --gpus "$worker_gpus" \
        --tp_size "$GPUS_PER_WORKER" \
        --mem_fraction_static "$MEM_FRACTION" \
        --batch_size "$BATCH_SIZE" \
        --seed "$worker_seed" \
        --output "$output_file" \
        --score_backend "$SCORE_BACKEND" \
        --timeout_score "$TIMEOUT_SCORE" \
        > "$log_file" 2>&1 &

    SLOT_PID[$slot_idx]=$!
    SLOT_REPEAT[$slot_idx]="$repeat_idx"
    SLOT_WORKER[$slot_idx]="$worker_idx"
    SLOT_START[$slot_idx]="$start_idx"
    SLOT_LIMIT[$slot_idx]="$worker_limit"
    SLOT_SEED[$slot_idx]="$worker_seed"
    SLOT_LOG[$slot_idx]="$log_file"
    SLOT_GPU[$slot_idx]="$worker_gpus"
    ACTIVE_JOBS=$((ACTIVE_JOBS + 1))
}

echo ""
echo "Dispatching jobs across $NUM_WORKERS worker slots..."

while [ "$NEXT_JOB_IDX" -lt "$TOTAL_JOBS" ] || [ "$ACTIVE_JOBS" -gt 0 ]; do
    for (( slot=0; slot<NUM_WORKERS; slot++ )); do
        if [ -z "${SLOT_PID[$slot]:-}" ] && [ "$NEXT_JOB_IDX" -lt "$TOTAL_JOBS" ]; then
            launch_job "$slot" "$NEXT_JOB_IDX"
            NEXT_JOB_IDX=$((NEXT_JOB_IDX + 1))
        fi
    done

    if [ "$ACTIVE_JOBS" -eq 0 ]; then
        break
    fi

    sleep 2

    for (( slot=0; slot<NUM_WORKERS; slot++ )); do
        pid="${SLOT_PID[$slot]:-}"
        if [ -z "$pid" ]; then
            continue
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            repeat_idx="${SLOT_REPEAT[$slot]}"
            worker_idx="${SLOT_WORKER[$slot]}"
            if wait "$pid"; then
                echo "Completed slot $slot: repeat $repeat_idx, worker $worker_idx"
            else
                echo "FAILED slot $slot: repeat $repeat_idx, worker $worker_idx. Check ${SLOT_LOG[$slot]}"
                FAILED=$((FAILED + 1))
            fi
            unset SLOT_PID[$slot] SLOT_REPEAT[$slot] SLOT_WORKER[$slot] SLOT_START[$slot] SLOT_LIMIT[$slot] SLOT_SEED[$slot] SLOT_LOG[$slot] SLOT_GPU[$slot]
            ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
        fi
    done
done

if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED worker job(s) failed."
fi

REPEAT_MERGED_FILES=()
for (( repeat_idx=1; repeat_idx<=REPEAT_COUNT; repeat_idx++ )); do
    MERGED_REPEAT_FILE="${RUN_DIR}/merged_results_repeat${repeat_idx}.jsonl"
    REPEAT_MERGED_FILES+=("$MERGED_REPEAT_FILE")
    cat "${RUN_DIR}"/worker_repeat${repeat_idx}_*.jsonl > "$MERGED_REPEAT_FILE" 2>/dev/null || true
done

BENCHMARK_END_TS=$(date +%s)
BENCHMARK_ELAPSED_SECONDS=$(( BENCHMARK_END_TS - BENCHMARK_START_TS ))

if [ "$REPEAT_COUNT" -gt 1 ]; then
    python3 "$AGGREGATE_SCRIPT" \
        --inputs "${REPEAT_MERGED_FILES[@]}" \
        --output_jsonl "${RUN_DIR}/merged_results.jsonl" \
        --summary_path "${RUN_DIR}/results.txt" \
        --mode direct \
        --benchmark "$BENCHMARK" \
        --model "$MODEL" \
        --max_tokens "$BENCH_MAX_TOKENS" \
        --repeat_seeds "${REPEAT_SEEDS[@]}" \
        --wall_clock_elapsed_seconds "$BENCHMARK_ELAPSED_SECONDS"
else
    cp "${REPEAT_MERGED_FILES[0]}" "${RUN_DIR}/merged_results.jsonl"

    python3 << PYEOF
import json, sys

results_file = "${RUN_DIR}/merged_results.jsonl"
summary_file = "${RUN_DIR}/results.txt"
benchmark = "${BENCHMARK}"
model = "${MODEL}"
max_tokens = "${BENCH_MAX_TOKENS}"
elapsed_seconds = float("${BENCHMARK_ELAPSED_SECONDS}")

total = correct = sum_think = sum_answer = 0
try:
    with open(results_file) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                total += 1
                correct += int(obj.get("correct", False))
                sum_think += obj.get("think_tokens", 0)
                sum_answer += obj.get("answer_tokens", 0)
except Exception as e:
    print(f"Error reading results: {e}")
    sys.exit(1)

if total > 0:
    acc = correct / total
    throughput = total / elapsed_seconds if elapsed_seconds > 0 else 0.0
    lines = [
        "=" * 70,
        "FINAL RESULTS (MERGED)",
        "=" * 70,
        f"Benchmark:         {benchmark}",
        f"Model:             {model}",
        f"Max tokens:        {max_tokens}",
        f"Total evaluated:   {total}",
        f"Accuracy:          {acc:.4f} ({correct}/{total})",
        f"Avg think tokens:  {sum_think / total:.1f}",
        f"Avg answer tokens: {sum_answer / total:.1f}",
        f"Avg total tokens:  {(sum_think + sum_answer) / total:.1f}",
        f"Elapsed:           {elapsed_seconds:.1f}s",
        f"Throughput:        {throughput:.2f} samples/s",
        "=" * 70,
        f"Merged results:    {results_file}",
        "=" * 70,
    ]
    text = "\n".join(lines)
    with open(summary_file, "w") as f:
        f.write(text + "\n")
    print(text)
else:
    msg = "No results found!"
    print(msg)
    with open(summary_file, "w") as f:
        f.write(msg + f"\nElapsed: {elapsed_seconds:.1f}s\n")
PYEOF
fi

echo "Benchmark $BENCHMARK done in ${BENCHMARK_ELAPSED_SECONDS}s"

done  # end benchmark loop

echo ""
OVERALL_END_TS=$(date +%s)
OVERALL_ELAPSED_SECONDS=$(( OVERALL_END_TS - OVERALL_START_TS ))
echo "All benchmarks done! Total elapsed: ${OVERALL_ELAPSED_SECONDS}s"
