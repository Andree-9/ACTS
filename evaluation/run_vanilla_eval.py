#!/usr/bin/env python3
"""
Direct model evaluation without controller.

Runs a reasoning model directly on benchmark problems using SGLang offline
Engine (consistent with slime eval which also uses SGLang).
Uses the model's own chat template via tokenizer.apply_chat_template().
Scores with math_verify (API-free) by default.

Supports parallel data splitting across GPUs via --start / --limit.

Usage (single worker):
    python evaluation/run_vanilla_eval.py \
        --model Qwen/Qwen3-8B \
        --benchmark math500 \
        --max_tokens 32768

Usage (called by run_vanilla_eval.sh for multi-GPU):
    python evaluation/run_vanilla_eval.py \
        --model Qwen/Qwen3-8B \
        --benchmark math500 \
        --start 0 --limit 63 \
        --gpus 0 \
        --output results/results_direct_generation/direct_math500_Qwen3-8B/worker_0.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root to path for evaluation module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import (
    DATASET_CONFIGS,
    extract_answer_candidate,
    score_response,
    count_think_answer_tokens,
    split_thinking_and_answer,
    load_benchmark,
)


def parse_args():
    p = argparse.ArgumentParser(description="Direct model evaluation (no controller)")
    p.add_argument("--model", type=str, required=True,
                   help="HuggingFace model name or local path")
    p.add_argument("--benchmark", type=str, required=True,
                   choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--start", type=int, default=0,
                   help="Start index for parallel splitting")
    p.add_argument("--limit", type=int, default=0, help="Max examples (0=all)")
    p.add_argument("--max_tokens", type=int, default=0,
                   help="Max generation tokens (0=dataset default)")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gpus", type=str, default=None,
                   help="Comma-separated GPU IDs")
    p.add_argument("--tp_size", type=int, default=1,
                   help="Tensor parallel size")
    p.add_argument("--mem_fraction_static", type=float, default=0.88,
                   help="SGLang static memory fraction")
    p.add_argument("--score_backend", type=str, default="math_verify",
                   choices=["exact", "math_verify"])
    p.add_argument("--timeout_score", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=-1,
                   help="Sampling seed (-1 = unset)")
    p.add_argument("--output", type=str, default="results/vanilla_eval_results.jsonl")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    done_ids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    try:
                        done_ids.add(json.loads(line)["doc_id"])
                    except Exception:
                        pass

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load benchmark (applies chat template to all prompts)
    samples = load_benchmark(
        args.benchmark,
        tokenizer=tokenizer,
        limit=None,
    )

    # Apply start/limit for parallel splitting
    if args.start > 0:
        samples = samples[args.start:]
    if args.limit > 0:
        samples = samples[: args.limit]
    if done_ids:
        samples = [s for s in samples if s.doc_id not in done_ids]

    cfg = DATASET_CONFIGS[args.benchmark]
    max_tokens = args.max_tokens if args.max_tokens > 0 else cfg["max_gen_toks"]

    print("=" * 70)
    print("Direct Model Evaluation (No Controller) — SGLang")
    print("=" * 70)
    print(f"Model:           {args.model}")
    print(f"Benchmark:       {args.benchmark}")
    print(f"Samples:         {len(samples)}")
    print(f"Max tokens:      {max_tokens}")
    print(f"Temperature:     {args.temperature}")
    print(f"Seed:            {args.seed if args.seed >= 0 else 'unset'}")
    print(f"Score backend:   {args.score_backend}")
    print(f"Output:          {args.output}")
    print("=" * 70)

    print("Loading model with SGLang Engine...")
    import sglang as sgl

    llm = sgl.Engine(
        model_path=args.model,
        tp_size=args.tp_size,
        mem_fraction_static=args.mem_fraction_static,
        trust_remote_code=True,
    )
    print("SGLang Engine ready.", flush=True)

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": max_tokens,
    }
    if args.top_k > 0:
        sampling_params["top_k"] = args.top_k

    output_mode = "a" if args.resume else "w"
    out_f = open(args.output, output_mode)
    correct = 0
    total = 0
    sum_think = 0
    sum_answer = 0
    t0 = time.time()

    try:
        for batch_start in tqdm(range(0, len(samples), args.batch_size),
                                desc=args.benchmark):
            batch = samples[batch_start : batch_start + args.batch_size]
            prompts = [s.formatted_prompt for s in batch]
            batch_sampling_params = dict(sampling_params)
            if args.seed >= 0:
                batch_sampling_params["sampling_seed"] = args.seed + batch_start

            print(f"[batch {batch_start}] Sending {len(prompts)} prompts to SGLang "
                  f"(prompt len ~{len(prompts[0])} chars, max_new_tokens={max_tokens})...",
                  flush=True)
            outputs = llm.generate(prompts, batch_sampling_params)
            print(f"[batch {batch_start}] Generation done.", flush=True)

            for sample, output in zip(batch, outputs):
                model_output = output["text"]
                think_tok, answer_tok = count_think_answer_tokens(
                    tokenizer,
                    model_output,
                    prompt_has_closed_think=False,
                )
                _, answer_text = split_thinking_and_answer(
                    model_output,
                    prompt_has_closed_think=False,
                )
                grading_output = answer_text if answer_text.strip() else model_output
                extracted = extract_answer_candidate(grading_output)
                is_correct = score_response(
                    grading_output, sample.answer,
                    backend=args.score_backend,
                    timeout_score=args.timeout_score,
                )

                total += 1
                correct += int(is_correct)
                sum_think += think_tok
                sum_answer += answer_tok

                if args.verbose:
                    status = "OK" if is_correct else "WRONG"
                    print(f"\n[{sample.doc_id}] {status} extracted='{extracted}' gt='{sample.answer}'")

                record = {
                    "doc_id": sample.doc_id,
                    "problem": sample.problem,
                    "ground_truth": sample.answer,
                    "correct": is_correct,
                    "extracted_answer": extracted,
                    "think_tokens": think_tok,
                    "answer_tokens": answer_tok,
                    "total_tokens": think_tok + answer_tok,
                    "seed": args.seed,
                    "model_output": model_output,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
    finally:
        out_f.close()
        llm.shutdown()

    elapsed = time.time() - t0
    acc = correct / total if total else 0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Evaluated:         {total}")
    print(f"Accuracy:          {acc:.4f} ({correct}/{total})")
    print(f"Avg think tokens:  {sum_think / total if total else 0:.1f}")
    print(f"Avg answer tokens: {sum_answer / total if total else 0:.1f}")
    print(f"Avg total tokens:  {(sum_think + sum_answer) / total if total else 0:.1f}")
    print(f"Elapsed:           {elapsed:.1f}s")
    print(f"Throughput:        {total / elapsed:.2f} samples/s")
    print("=" * 70)


if __name__ == "__main__":
    main()
