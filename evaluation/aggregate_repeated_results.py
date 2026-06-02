#!/usr/bin/env python3
"""
Aggregate repeated evaluation runs into a single averaged result file.

For repeated benchmarks, the canonical merged file stores per-sample averages
across repeats.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate repeated eval results")
    p.add_argument("--inputs", nargs="+", required=True, help="Per-repeat JSONL files")
    p.add_argument("--output_jsonl", required=True, help="Aggregated merged_results.jsonl path")
    p.add_argument("--summary_path", required=True, help="Aggregated results.txt path")
    p.add_argument("--mode", choices=["direct", "controlled"], required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--model", default="")
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--repeat_seeds", nargs="*", type=int, default=[])
    p.add_argument("--repeat_elapsed_seconds", nargs="*", type=float, default=[])
    p.add_argument("--wall_clock_elapsed_seconds", type=float, default=None)
    return p.parse_args()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))


def _load_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: row["doc_id"])
    return rows


def _build_aggregated_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    correct_values = [float(bool(row.get("correct", False))) for row in rows]
    think_values = [float(row.get("think_tokens", 0.0)) for row in rows]
    answer_values = [float(row.get("answer_tokens", 0.0)) for row in rows]
    total_values = [
        float(row.get("total_tokens", row.get("think_tokens", 0.0) + row.get("answer_tokens", 0.0)))
        for row in rows
    ]

    aggregated = {
        "doc_id": first["doc_id"],
        "problem": first.get("problem"),
        "ground_truth": first.get("ground_truth"),
        "correct": _mean(correct_values),
        "correct_std": _std(correct_values),
        "think_tokens": _mean(think_values),
        "think_tokens_std": _std(think_values),
        "answer_tokens": _mean(answer_values),
        "answer_tokens_std": _std(answer_values),
        "total_tokens": _mean(total_values),
        "total_tokens_std": _std(total_values),
        "n_repeats": len(rows),
        "is_aggregated": True,
        "repeat_correct": correct_values,
        "repeat_think_tokens": think_values,
        "repeat_answer_tokens": answer_values,
        "repeat_total_tokens": total_values,
    }
    if "extracted_answer" in first:
        aggregated["repeat_extracted_answers"] = [row.get("extracted_answer", "") for row in rows]
    if "n_steps" in first:
        step_values = [float(row.get("n_steps", 0.0)) for row in rows]
        aggregated["n_steps"] = _mean(step_values)
        aggregated["n_steps_std"] = _std(step_values)
        aggregated["repeat_n_steps"] = step_values

    return aggregated


def _summarize_run(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = len(rows)
    correct = sum(float(bool(row.get("correct", False))) for row in rows)
    sum_think = sum(float(row.get("think_tokens", 0.0)) for row in rows)
    sum_answer = sum(float(row.get("answer_tokens", 0.0)) for row in rows)
    summary = {
        "total": float(total),
        "accuracy": (correct / total) if total else 0.0,
        "avg_think": (sum_think / total) if total else 0.0,
        "avg_answer": (sum_answer / total) if total else 0.0,
        "avg_total": ((sum_think + sum_answer) / total) if total else 0.0,
    }
    if total and "n_steps" in rows[0]:
        summary["avg_steps"] = sum(float(row.get("n_steps", 0.0)) for row in rows) / total
    return summary


def main() -> None:
    args = parse_args()
    repeat_rows = [_load_rows(path) for path in args.inputs]
    if not repeat_rows:
        raise ValueError("No input files provided")

    expected_doc_ids = [row["doc_id"] for row in repeat_rows[0]]
    for path, rows in zip(args.inputs[1:], repeat_rows[1:], strict=False):
        doc_ids = [row["doc_id"] for row in rows]
        if doc_ids != expected_doc_ids:
            raise ValueError(f"Doc IDs do not match across repeats: {path}")

    aggregated_rows = [
        _build_aggregated_row(list(row_group))
        for row_group in zip(*repeat_rows, strict=True)
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)) or ".", exist_ok=True)
    with open(args.output_jsonl, "w") as f:
        for row in aggregated_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    run_summaries = [_summarize_run(rows) for rows in repeat_rows]
    total = len(aggregated_rows)
    mean_accuracy = _mean([s["accuracy"] for s in run_summaries])
    std_accuracy = _std([s["accuracy"] for s in run_summaries])
    mean_think = _mean([s["avg_think"] for s in run_summaries])
    std_think = _std([s["avg_think"] for s in run_summaries])
    mean_answer = _mean([s["avg_answer"] for s in run_summaries])
    std_answer = _std([s["avg_answer"] for s in run_summaries])
    mean_total = _mean([s["avg_total"] for s in run_summaries])
    std_total = _std([s["avg_total"] for s in run_summaries])
    mean_elapsed = _mean(args.repeat_elapsed_seconds)
    total_elapsed = sum(args.repeat_elapsed_seconds)
    mean_throughput = (
        _mean([
            summary["total"] / elapsed
            for summary, elapsed in zip(run_summaries, args.repeat_elapsed_seconds, strict=False)
            if elapsed > 0
        ])
        if args.repeat_elapsed_seconds
        else 0.0
    )
    wall_clock_elapsed = args.wall_clock_elapsed_seconds
    raw_total = total * len(repeat_rows)
    wall_clock_throughput = (
        raw_total / wall_clock_elapsed
        if wall_clock_elapsed is not None and wall_clock_elapsed > 0
        else 0.0
    )

    lines = [
        "=" * 70,
        "FINAL RESULTS (AGGREGATED OVER REPEATS)",
        "=" * 70,
        f"Benchmark:         {args.benchmark}",
    ]
    if args.mode == "direct" and args.model:
        lines.append(f"Model:             {args.model}")
    if args.mode == "direct" and args.max_tokens is not None:
        lines.append(f"Max tokens:        {args.max_tokens}")
    if args.mode == "controlled" and args.budget is not None:
        lines.append(f"Budget:            {args.budget}")
    lines.extend([
        f"Repeats:           {len(repeat_rows)}",
        f"Total evaluated:   {total}",
        f"Accuracy:          {mean_accuracy:.4f} +/- {std_accuracy:.4f}",
        f"Avg think tokens:  {mean_think:.1f} +/- {std_think:.1f}",
        f"Avg answer tokens: {mean_answer:.1f} +/- {std_answer:.1f}",
        f"Avg total tokens:  {mean_total:.1f} +/- {std_total:.1f}",
    ])
    if args.mode == "controlled":
        step_values = [s.get("avg_steps", 0.0) for s in run_summaries]
        lines.append(f"Avg steps:         {_mean(step_values):.1f} +/- {_std(step_values):.1f}")
    if wall_clock_elapsed is not None:
        lines.extend([
            f"Elapsed:           {wall_clock_elapsed:.1f}s",
            f"Throughput:        {wall_clock_throughput:.2f} samples/s",
        ])
    elif args.repeat_elapsed_seconds:
        lines.extend([
            f"Avg elapsed:       {mean_elapsed:.1f}s",
            f"Total elapsed:     {total_elapsed:.1f}s",
            f"Avg throughput:    {mean_throughput:.2f} samples/s",
        ])
    if args.repeat_seeds:
        seed_text = ", ".join(str(seed) for seed in args.repeat_seeds)
        lines.append(f"Repeat seeds:      {seed_text}")
    lines.extend([
        "=" * 70,
        f"Merged results:    {args.output_jsonl}",
        "=" * 70,
    ])

    summary = "\n".join(lines)
    with open(args.summary_path, "w") as f:
        f.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
