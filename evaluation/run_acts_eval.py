#!/usr/bin/env python3
"""
ACTS Evaluation — standalone async SGLang version.

Evaluates the controller-reasoner pipeline on reasoning benchmarks by talking
to two SGLang HTTP servers (controller + reasoner) via async HTTP requests.
All samples are processed concurrently for high throughput.

The multi-turn loop mirrors acts_rollout.py:
  1. Controller receives question + budget → outputs action + starter
  2. Reasoner generates a reasoning step using the starter
  3. Environment feedback (step + budget %) sent back to controller
  4. Repeat until </think> or max turns
  5. Final answer generation: reasoner continues after </think>

Usage:
    python evaluation/run_acts_eval.py \
        --controller_url http://localhost:30000 \
        --reasoner_url http://localhost:30001 \
        --controller_model path/to/controller \
        --reasoner_model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --benchmark math500 \
        --budget 3511
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from typing import Any, Optional

import warnings
warnings.filterwarnings("ignore")

import httpx
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import (
    DATASET_CONFIGS,
    extract_answer_candidate,
    score_response,
    load_benchmark,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (same as acts_rollout.py)
# ---------------------------------------------------------------------------

CONTROL_AGENT_SYSTEM_PROMPT = """You are a reasoning control agent that guides a step-by-step mathematical reasoner.

Each turn you receive the reasoner's latest step and the remaining token budget (as a percentage). On the first turn you receive the question instead. Considering the reasoner's current progress and planning on the remaining budget, choose an action and a starter phrase for the reasoner's next step. Your goal is to guide the reasoner to make progress towards the final correct answer while staying within the budget.

Actions:
- understand: Restate or clarify the problem.
- plan: Decide strategy or split into subproblems.
- execute: Carry out algebraic or logical steps.
- explore: Try an alternative approach.
- check: Verify steps or correct mistakes.
- summarize: Summarize current results.
- conclude: State the final answer.

Output format:
Action: <action>
Starter: <phrase that steers the reasoner toward the chosen action>"""

ALLOWED_ACTIONS = {
    "understand", "plan", "execute", "explore", "check", "summarize", "conclude",
}

DEFAULT_STOP_SEQUENCES = [".\n\n", "?\n\n", ". \n\n", "? \n\n", "</think>"]
FORCED_CONCLUDE_STARTER = "\n\n**Final Answer**\n"

ACTION_PATTERN = re.compile(r"Action\s*:\s*([A-Za-z_]+)", flags=re.IGNORECASE)
STARTER_PATTERN = re.compile(r"Starter\s*:\s*(.+?)(?:\n|$)", flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_action_output(action_text: str) -> tuple[str, str]:
    """Parse action and starter from controller output."""
    action_name = None
    starter = ""

    action_match = ACTION_PATTERN.search(action_text or "")
    if action_match:
        candidate = action_match.group(1).strip().lower()
        if candidate in ALLOWED_ACTIONS:
            action_name = candidate

    if action_name is None:
        lowered = (action_text or "").lower()
        for action in ALLOWED_ACTIONS:
            if re.search(rf"\b{re.escape(action)}\b", lowered):
                action_name = action
                break

    starter_match = STARTER_PATTERN.search(action_text or "")
    if starter_match:
        starter = starter_match.group(1).strip()
        if (starter.startswith('"') and starter.endswith('"')) or (
            starter.startswith("'") and starter.endswith("'")
        ):
            starter = starter[1:-1]

    if action_name is None or not starter:
        return "execute", "None"

    return action_name, starter


def _build_reasoner_prompt(question: str, reasoning_trace: str, starter: str, tokenizer) -> str:
    """Build the reasoner prompt using the model's own chat template.

    Uses tokenizer.apply_chat_template for the base prompt (user message +
    generation prompt), then appends the <think> continuation manually.

    Some model templates (e.g., DeepSeek-R1) already include <think> in their
    generation prompt. We strip it to avoid duplication.
    """
    messages = [{"role": "user", "content": question}]
    base_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    # Strip trailing <think> if the template already added it
    base_prompt = base_prompt.removesuffix("<think>\n").removesuffix("<think>")
    if reasoning_trace:
        return base_prompt + f"<think>\n{reasoning_trace}{starter}"
    else:
        return base_prompt + f"<think>\n{starter}"


def _route_rank(route_key: str, dp_size: int) -> int | None:
    """Map a sample to a stable native-DP rank for per-sample cache locality."""
    if dp_size <= 1:
        return None
    digest = hashlib.blake2b(route_key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dp_size


def _with_routed_dp_rank(payload: dict, rank: int | None) -> dict:
    if rank is not None:
        payload["routed_dp_rank"] = rank
    return payload


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def _post(
    client: httpx.AsyncClient, url: str, payload: dict,
    max_retries: int = 30,
) -> dict:
    """POST with retry, matching slime's http_utils pattern."""
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt >= max_retries:
                raise
            logger.warning(f"POST {url} failed (attempt {attempt}): {e}")
            await asyncio.sleep(1)


# ---------------------------------------------------------------------------
# Multi-turn loop for a single problem
# ---------------------------------------------------------------------------

async def _run_one_problem(
    tracked_post,
    controller_url: str,
    reasoner_url: str,
    ctrl_tokenizer: AutoTokenizer,
    budget_tokenizer: AutoTokenizer,
    problem: str,
    budget: int,
    args: argparse.Namespace,
    sample_seed: int,
    controller_route_rank: int | None = None,
    reasoner_route_rank: int | None = None,
) -> dict[str, Any]:
    """Run the controller-reasoner multi-turn loop for one problem."""

    messages = [
        {"role": "system", "content": CONTROL_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question:\n{problem}\n\nBudget Remaining: 100%"},
    ]
    prompt_str = ctrl_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    input_ids = ctrl_tokenizer.encode(prompt_str, add_special_tokens=False)

    ctrl_eos_id = ctrl_tokenizer.convert_tokens_to_ids("<|im_end|>")

    reasoning_trace = ""
    real_budget_remaining = float(budget)
    step_idx = 0
    actions_log = []
    starters_log = []
    budget_pcts_log = []
    done = False
    trace_cap_reached = False
    reasoning_trace_tokens = 0

    for _turn in range(args.max_turns):
        if done:
            break

        # ===== A. Controller call =====
        turn_seed_base = sample_seed + (_turn * 4) if sample_seed >= 0 else -1
        ctrl_sampling_params = {
            "temperature": args.controller_temperature,
            "top_p": args.controller_top_p,
            "max_new_tokens": 256,
            "stop_token_ids": [ctrl_eos_id],
        }
        if turn_seed_base >= 0:
            ctrl_sampling_params["sampling_seed"] = turn_seed_base
        ctrl_payload = _with_routed_dp_rank({
            "input_ids": input_ids,
            "sampling_params": ctrl_sampling_params,
            "return_logprob": True,
        }, controller_route_rank)
        ctrl_output = await tracked_post(controller_url, ctrl_payload, kind="ctrl")
        ctrl_text = ctrl_output["text"]
        ctrl_meta = ctrl_output.get("meta_info", {})
        ctrl_tokens = [item[1] for item in ctrl_meta.get("output_token_logprobs", [])]
        input_ids.extend(ctrl_tokens)

        action_name, action_starter = _parse_action_output(ctrl_text)
        budget_pct = max(0, int(100 * real_budget_remaining / float(budget)))
        actions_log.append(action_name)
        starters_log.append(action_starter)
        budget_pcts_log.append(budget_pct)

        # ===== B. Reasoner call =====
        starter_for_reasoner = ""
        if action_starter and action_starter.strip().lower() != "none":
            starter_for_reasoner = action_starter

        reasoner_prompt = _build_reasoner_prompt(
            problem, reasoning_trace, starter_for_reasoner, budget_tokenizer,
        )
        reasoner_sampling_params = {
            "temperature": args.reasoner_temperature,
            "top_p": args.reasoner_top_p,
            "max_new_tokens": args.reasoner_max_tokens_per_step,
            "stop": list(DEFAULT_STOP_SEQUENCES),
            "no_stop_trim": True,
        }
        if turn_seed_base >= 0:
            reasoner_sampling_params["sampling_seed"] = turn_seed_base + 1
        reasoner_payload = _with_routed_dp_rank({
            "text": reasoner_prompt,
            "sampling_params": reasoner_sampling_params,
        }, reasoner_route_rank)
        reasoner_output = await tracked_post(
            reasoner_url, reasoner_payload, kind="reasoner",
        )
        reasoner_text = reasoner_output["text"]
        finish_type = reasoner_output.get("meta_info", {}).get("finish_reason", {}).get("type", "length")

        step_text = starter_for_reasoner + reasoner_text
        step_done = False

        if finish_type == "stop":
            if "</think>" in reasoner_text or step_text.rstrip().endswith("</think>"):
                step_done = True
                if not step_text.rstrip().endswith("</think>"):
                    step_text += "</think>"
        if "</think>" in step_text:
            step_done = True

        reasoning_trace += step_text

        budget_text = step_text.replace("</think>", "").strip()
        tokens_used = len(budget_tokenizer.encode(budget_text, add_special_tokens=False))
        real_budget_remaining -= float(tokens_used)
        reasoning_trace_tokens += tokens_used
        step_idx += 1

        if (
            args.max_reasoning_trace_tokens > 0
            and reasoning_trace_tokens >= args.max_reasoning_trace_tokens
            and not step_done
        ):
            trace_cap_reached = True
            break

        # ===== C. Forced conclude =====
        if action_name == "conclude" and not step_done:
            forced_prompt = _build_reasoner_prompt(
                problem, reasoning_trace, FORCED_CONCLUDE_STARTER, budget_tokenizer,
            )
            forced_params = {
                "temperature": args.reasoner_temperature,
                "top_p": args.reasoner_top_p,
                "max_new_tokens": args.forced_conclude_max_tokens,
                "stop": ["</think>"],
                "no_stop_trim": True,
            }
            if turn_seed_base >= 0:
                forced_params["sampling_seed"] = turn_seed_base + 2
            forced_payload = _with_routed_dp_rank({
                "text": forced_prompt,
                "sampling_params": forced_params,
            }, reasoner_route_rank)
            forced_output = await tracked_post(
                reasoner_url, forced_payload, kind="reasoner",
            )
            forced_text = forced_output["text"]
            forced_finish = forced_output.get("meta_info", {}).get("finish_reason", {}).get("type", "length")
            forced_text = FORCED_CONCLUDE_STARTER + forced_text
            if forced_finish == "stop" and not forced_text.rstrip().endswith("</think>"):
                forced_text += "</think>"
            if not forced_text.rstrip().endswith("</think>"):
                forced_text = forced_text.rstrip() + "\n</think>"

            reasoning_trace += forced_text
            step_text += forced_text
            step_done = True

            forced_budget_text = forced_text.replace("</think>", "").strip()
            forced_tokens = len(budget_tokenizer.encode(forced_budget_text, add_special_tokens=False))
            real_budget_remaining -= float(forced_tokens)
            reasoning_trace_tokens += forced_tokens

        done = step_done or action_name == "conclude"

        # ===== D. Environment feedback (build next controller input) =====
        if not done:
            budget_pct = max(0, int(100 * real_budget_remaining / float(budget)))
            step_for_prompt = step_text.strip() if step_text.strip() else "<no step content>"
            env_content = f"Reasoner's Step:\n{step_for_prompt}\n\nBudget Remaining: {budget_pct}%"
            env_messages = [{"role": "user", "content": env_content}]
            env_ids = ctrl_tokenizer.apply_chat_template(
                env_messages, tokenize=True, add_generation_prompt=True,
            )
            # Strip leading BOS (mid-sequence append)
            bos_id = ctrl_tokenizer.bos_token_id
            if bos_id is not None and env_ids and env_ids[0] == bos_id:
                env_ids = env_ids[1:]
            # Prepend \n to connect after controller's <|im_end|>
            newline_ids = ctrl_tokenizer.encode("\n", add_special_tokens=False)
            env_ids = newline_ids + env_ids
            input_ids.extend(env_ids)

        if done:
            break

    # ===== E. Final answer generation =====
    answer_text = ""
    if done:
        trace = reasoning_trace.rstrip()
        if not trace.endswith("</think>"):
            trace += "\n</think>"
        final_messages = [{"role": "user", "content": problem}]
        final_prompt = budget_tokenizer.apply_chat_template(
            final_messages, tokenize=False, add_generation_prompt=True,
        )
        final_prompt = final_prompt.removesuffix("<think>\n").removesuffix("<think>")
        final_prompt += f"<think>\n{trace}\n\n"
        final_params = {
            "temperature": args.reasoner_temperature,
            "top_p": args.reasoner_top_p,
            "max_new_tokens": args.final_answer_max_tokens,
        }
        if sample_seed >= 0:
            final_params["sampling_seed"] = sample_seed + (args.max_turns * 4) + 3
        final_payload = {
            "text": final_prompt,
            "sampling_params": final_params,
        }
        _with_routed_dp_rank(final_payload, reasoner_route_rank)
        final_output = await tracked_post(reasoner_url, final_payload, kind="reasoner")
        answer_text = final_output["text"]

    think_tokens = int(budget - real_budget_remaining) if budget > 0 else 0

    return {
        "reasoning_trace": reasoning_trace,
        "answer_text": answer_text,
        "think_tokens": think_tokens,
        "reasoning_trace_tokens": reasoning_trace_tokens,
        "trace_cap_reached": trace_cap_reached,
        "max_reasoning_trace_tokens": args.max_reasoning_trace_tokens,
        "n_steps": step_idx,
        "actions": actions_log,
        "starters": starters_log,
        "budget_remaining_pcts": budget_pcts_log,
        "real_budget_remaining": real_budget_remaining,
    }


# ---------------------------------------------------------------------------
# Concurrency monitor
# ---------------------------------------------------------------------------

class AsyncStats:
    """Lightweight counters for monitoring async throughput."""

    def __init__(self):
        self.active_samples = 0      # samples currently in the multi-turn loop
        self.active_requests = 0     # HTTP requests currently in flight
        self.total_requests = 0      # total HTTP requests sent
        self.total_ctrl_calls = 0    # controller calls
        self.total_reasoner_calls = 0  # reasoner calls
        self.completed_samples = 0
        # Per-role inflight & latency tracking
        self.inflight_ctrl = 0
        self.inflight_reas = 0
        self.ctrl_total_latency = 0.0
        self.reas_total_latency = 0.0
        self._lock = asyncio.Lock()

    async def sample_start(self):
        async with self._lock:
            self.active_samples += 1

    async def sample_done(self):
        async with self._lock:
            self.active_samples -= 1
            self.completed_samples += 1

    async def request_start(self, kind: str = ""):
        async with self._lock:
            self.active_requests += 1
            self.total_requests += 1
            if kind == "ctrl":
                self.total_ctrl_calls += 1
                self.inflight_ctrl += 1
            elif kind == "reasoner":
                self.total_reasoner_calls += 1
                self.inflight_reas += 1

    async def request_done(self, kind: str = "", latency: float = 0.0):
        async with self._lock:
            self.active_requests -= 1
            if kind == "ctrl":
                self.inflight_ctrl -= 1
                self.ctrl_total_latency += latency
            elif kind == "reasoner":
                self.inflight_reas -= 1
                self.reas_total_latency += latency

    def snapshot(self) -> dict:
        ctrl_avg = (self.ctrl_total_latency / self.total_ctrl_calls
                    if self.total_ctrl_calls else 0.0)
        reas_avg = (self.reas_total_latency / self.total_reasoner_calls
                    if self.total_reasoner_calls else 0.0)
        return {
            "active": self.active_samples,
            "inflight": self.active_requests,
            "reqs": self.total_requests,
            "ctrl": self.total_ctrl_calls,
            "reas": self.total_reasoner_calls,
            "ctrl_q": self.inflight_ctrl,
            "reas_q": self.inflight_reas,
            "ctrl_avg_s": round(ctrl_avg, 2),
            "reas_avg_s": round(reas_avg, 2),
        }


# ---------------------------------------------------------------------------
# Async evaluation driver
# ---------------------------------------------------------------------------

async def eval_all(
    samples: list,
    args: argparse.Namespace,
    ctrl_tokenizer: AutoTokenizer,
    budget_tokenizer: AutoTokenizer,
) -> list[dict]:
    """Evaluate all samples with bounded concurrency."""

    controller_url = args.controller_url.rstrip("/") + "/generate"
    reasoner_url = args.reasoner_url.rstrip("/") + "/generate"

    semaphore = asyncio.Semaphore(args.concurrency)
    stats = AsyncStats()

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=args.concurrency * 4),
        timeout=httpx.Timeout(None),
    ) as client:

        async def tracked_post(url, payload, kind=""):
            await stats.request_start(kind)
            t0 = time.monotonic()
            try:
                return await _post(client, url, payload)
            finally:
                await stats.request_done(kind, time.monotonic() - t0)

        async def process_one(sample) -> dict:
            async with semaphore:
                await stats.sample_start()
                route_key = f"{args.benchmark}:{sample.doc_id}"
                controller_route_rank = _route_rank(
                    f"controller:{route_key}", args.controller_dp,
                )
                reasoner_route_rank = _route_rank(
                    f"reasoner:{route_key}", args.reasoner_dp,
                )
                try:
                    episode = await _run_one_problem(
                        tracked_post=tracked_post,
                        controller_url=controller_url,
                        reasoner_url=reasoner_url,
                        ctrl_tokenizer=ctrl_tokenizer,
                        budget_tokenizer=budget_tokenizer,
                        problem=sample.problem,
                        budget=args.budget,
                        args=args,
                        sample_seed=(args.seed + sample.doc_id * 10000) if args.seed >= 0 else -1,
                        controller_route_rank=controller_route_rank,
                        reasoner_route_rank=reasoner_route_rank,
                    )
                finally:
                    await stats.sample_done()

                grading_output = episode["answer_text"] or episode["reasoning_trace"]
                extracted = extract_answer_candidate(grading_output)
                is_correct = score_response(
                    grading_output, sample.answer,
                    backend=args.score_backend,
                    timeout_score=args.timeout_score,
                )

                answer_tokens = len(budget_tokenizer.encode(
                    episode["answer_text"], add_special_tokens=False,
                )) if episode["answer_text"] else 0

                return {
                    "doc_id": sample.doc_id,
                    "problem": sample.problem,
                    "ground_truth": sample.answer,
                    "correct": is_correct,
                    "extracted_answer": extracted,
                    "think_tokens": episode["think_tokens"],
                    "answer_tokens": answer_tokens,
                    "n_steps": episode["n_steps"],
                    "actions": episode["actions"],
                    "starters": episode["starters"],
                    "budget_remaining_pcts": episode["budget_remaining_pcts"],
                    "reasoning_trace": episode["reasoning_trace"],
                    "answer_text": episode["answer_text"],
                    "seed": args.seed,
                    "controller_dp_rank": controller_route_rank,
                    "reasoner_dp_rank": reasoner_route_rank,
                }

        # Launch all tasks concurrently (bounded by semaphore)
        tasks = [asyncio.create_task(process_one(s)) for s in samples]

        results = []
        pbar = tqdm(total=len(tasks), desc=args.benchmark)
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            snap = stats.snapshot()
            pbar.set_postfix_str(
                f"active={snap['active']} "
                f"ctrl(q={snap['ctrl_q']},avg={snap['ctrl_avg_s']}s) "
                f"reas(q={snap['reas_q']},avg={snap['reas_avg_s']}s)"
            )
            pbar.update(1)
        pbar.close()

    results.sort(key=lambda r: r["doc_id"])
    return results, stats.snapshot()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="ACTS Evaluation (standalone async SGLang)",
    )

    # Server URLs
    p.add_argument("--controller_url", type=str, default="http://localhost:30000",
                   help="Controller SGLang server URL")
    p.add_argument("--reasoner_url", type=str, default="http://localhost:30001",
                   help="Reasoner SGLang server URL")
    p.add_argument("--controller_dp", type=int, default=1,
                   help="Controller native DP size; used for sticky routed_dp_rank")
    p.add_argument("--reasoner_dp", type=int, default=1,
                   help="Reasoner native DP size; used for sticky routed_dp_rank")

    # Model paths (for tokenizer loading only — inference is via HTTP servers)
    p.add_argument("--controller_model", type=str, required=True,
                   help="Controller model path (for tokenizer)")
    p.add_argument("--reasoner_model", type=str,
                   default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                   help="Reasoner model path (for tokenizer)")

    # Dataset
    p.add_argument("--benchmark", type=str, required=True,
                   choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="Max examples (0=all)")

    # Budget/control
    p.add_argument("--budget", type=int, required=True,
                   help="Token budget for thinking")
    p.add_argument("--max_turns", type=int, default=100)
    p.add_argument("--max_reasoning_trace_tokens", type=int, default=32768,
                   help="Hard cap on generated reasoning trace tokens before truncating")

    # Controller generation
    p.add_argument("--controller_temperature", type=float, default=0.7)
    p.add_argument("--controller_top_p", type=float, default=0.8)

    # Reasoner generation
    p.add_argument("--reasoner_temperature", type=float, default=0.6)
    p.add_argument("--reasoner_top_p", type=float, default=0.95)
    p.add_argument("--reasoner_max_tokens_per_step", type=int, default=512)
    p.add_argument("--forced_conclude_max_tokens", type=int, default=256)
    p.add_argument("--final_answer_max_tokens", type=int, default=2048)

    # Scoring
    p.add_argument("--score_backend", type=str, default="math_verify",
                   choices=["exact", "math_verify"])
    p.add_argument("--timeout_score", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=-1,
                   help="Base sampling seed (-1 = unset)")

    # Concurrency
    p.add_argument("--concurrency", type=int, default=64,
                   help="Max concurrent samples in flight")

    # Output
    p.add_argument("--output_dir", type=str, default="results/results_controlled_eval",
                   help="Output directory")
    p.add_argument("--output_suffix", type=str, default="",
                   help="Optional filename suffix, e.g. _repeat1")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading controller tokenizer: {args.controller_model}")
    ctrl_tokenizer = AutoTokenizer.from_pretrained(
        args.controller_model, trust_remote_code=True,
    )
    print(f"Loading reasoner tokenizer: {args.reasoner_model}")
    budget_tokenizer = AutoTokenizer.from_pretrained(
        args.reasoner_model, trust_remote_code=True,
    )

    # Load benchmark (no chat template needed — controller has its own prompt format)
    samples = load_benchmark(args.benchmark, tokenizer=None, limit=None)

    if args.start > 0:
        samples = samples[args.start:]
    if args.limit > 0:
        samples = samples[:args.limit]

    run_dir = os.path.join(
        args.output_dir,
        f"controlled_{args.benchmark}_budget{args.budget}",
    )
    os.makedirs(run_dir, exist_ok=True)
    suffix = args.output_suffix or ""
    jsonl_path = os.path.join(run_dir, f"merged_results{suffix}.jsonl")
    summary_path = os.path.join(run_dir, f"results{suffix}.txt")

    done_ids = set()
    if args.resume and os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    try:
                        done_ids.add(json.loads(line)["doc_id"])
                    except Exception:
                        pass
        samples = [s for s in samples if s.doc_id not in done_ids]

    print("=" * 70)
    print("ACTS Evaluation (Async SGLang)")
    print("=" * 70)
    print(f"Benchmark:         {args.benchmark}")
    print(f"Samples:           {len(samples)}")
    print(f"Budget:            {args.budget}")
    print(f"Max turns:         {args.max_turns}")
    print(f"Concurrency:       {args.concurrency}")
    print(f"Seed:              {args.seed if args.seed >= 0 else 'unset'}")
    print(f"Controller URL:    {args.controller_url}")
    print(f"Reasoner URL:      {args.reasoner_url}")
    print(f"Score backend:     {args.score_backend}")
    print(f"Output:            {jsonl_path}")
    print("=" * 70)

    t0 = time.time()
    results, final_stats = asyncio.run(eval_all(samples, args, ctrl_tokenizer, budget_tokenizer))
    elapsed = time.time() - t0

    mode = "a" if args.resume and done_ids else "w"
    with open(jsonl_path, mode) as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    sum_think = sum(r["think_tokens"] for r in results)
    sum_answer = sum(r["answer_tokens"] for r in results)
    sum_steps = sum(r["n_steps"] for r in results)

    if total > 0:
        acc = correct / total
        lines = [
            "=" * 70,
            "FINAL RESULTS",
            "=" * 70,
            f"Benchmark:         {args.benchmark}",
            f"Budget:            {args.budget}",
            f"Total evaluated:   {total}",
            f"Accuracy:          {acc:.4f} ({correct}/{total})",
            f"Avg think tokens:  {sum_think / total:.1f}",
            f"Avg answer tokens: {sum_answer / total:.1f}",
            f"Avg total tokens:  {(sum_think + sum_answer) / total:.1f}",
            f"Avg steps:         {sum_steps / total:.1f}",
            f"Elapsed:           {elapsed:.1f}s",
            f"Throughput:        {total / elapsed:.2f} samples/s",
            "-" * 70,
            "ENGINE UTILIZATION (for GPU re-allocation)",
            f"  Controller:      {final_stats['ctrl']} calls, avg {final_stats['ctrl_avg_s']:.2f}s/call",
            f"  Reasoner:        {final_stats['reas']} calls, avg {final_stats['reas_avg_s']:.2f}s/call",
            f"  Bottleneck:      {'reasoner' if final_stats['reas_avg_s'] > final_stats['ctrl_avg_s'] else 'controller'} (higher avg latency)",
            "=" * 70,
            f"Results:           {jsonl_path}",
            "=" * 70,
        ]
        summary = "\n".join(lines)
        with open(summary_path, "w") as f:
            f.write(summary + "\n")
        print(f"\n{summary}")
    else:
        print("No results.")


if __name__ == "__main__":
    main()
