"""
Custom generate function for ACTS controller RL from full prompts.

Wired via: --custom-generate-function-path slime.rollout.acts_rollout.generate_group

Full-continuation GRPO formulation:
  - sample.prompt is the start controller dialog: system plus the question user
    turn with "Budget Remaining: 100%".
  - Each GRPO sample rolls out a complete controller/reasoner continuation
    from the question until terminal, trace cap, response cap, or max turns.
  - All generated controller turns in the continuation are trainable
    (loss_mask=1). Controller-side user feedback containing reasoner steps and
    updated budget text is appended to the model input with loss_mask=0.
  - The terminal answer from the continuation provides the scalar reward for
    the whole controller trajectory.
  - Budget accounting keeps two separate values:
      * metadata["initial_budget_remaining"] starts the continuation.
      * metadata["original_total_budget"] is the denominator for every later
        "Budget Remaining: XX%" feedback message.
    For full-prompt RL these values must match, so the first controller turn
    starts at 100% remaining under the sampled total budget.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Optional

from transformers import AutoTokenizer

from slime.rollout.sglang_rollout import GenerateState, get_model_url
from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rollout concurrency monitor (module-level singleton)
# ---------------------------------------------------------------------------


class RolloutStats:
    """Tracks active samples and inflight HTTP requests during training rollout."""

    _instance: Optional["RolloutStats"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._lock = asyncio.Lock()
        self.active_samples = 0
        self.active_requests = 0
        self.total_requests = 0
        self.total_ctrl_calls = 0
        self.total_reasoner_calls = 0
        self.completed_samples = 0
        self._last_log_time = 0.0
        self._log_interval = 10.0

    async def sample_start(self):
        async with self._lock:
            self.active_samples += 1

    async def sample_done(self):
        async with self._lock:
            self.active_samples -= 1
            self.completed_samples += 1
            self._maybe_log()

    async def request_start(self, kind: str = ""):
        async with self._lock:
            self.active_requests += 1
            self.total_requests += 1
            if kind == "ctrl":
                self.total_ctrl_calls += 1
            elif kind == "reasoner":
                self.total_reasoner_calls += 1

    async def request_done(self):
        async with self._lock:
            self.active_requests -= 1
            self._maybe_log()

    def _maybe_log(self):
        now = time.monotonic()
        if now - self._last_log_time >= self._log_interval:
            self._last_log_time = now
            logger.info(
                f"[acts_rollout] active={self.active_samples} "
                f"inflight={self.active_requests} "
                f"reqs={self.total_requests} "
                f"ctrl={self.total_ctrl_calls} "
                f"reas={self.total_reasoner_calls} "
                f"done={self.completed_samples}"
            )


_rollout_stats = RolloutStats()


async def _tracked_post(
    url: str, payload: dict, kind: str = "", headers: dict | None = None,
) -> dict:
    await _rollout_stats.request_start(kind)
    try:
        return await post(url, payload, headers=headers)
    finally:
        await _rollout_stats.request_done()


# Cached reasoner tokenizer (for budget accounting).
_reasoner_tokenizer = None


def _get_reasoner_tokenizer(model_path: str):
    global _reasoner_tokenizer
    if _reasoner_tokenizer is None:
        _reasoner_tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
    return _reasoner_tokenizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


ALLOWED_ACTIONS = {
    "understand", "plan", "execute", "explore", "check", "summarize", "conclude",
}

DEFAULT_STOP_SEQUENCES = [".\n\n", "?\n\n", ". \n\n", "? \n\n", "</think>"]
FORCED_CONCLUDE_STARTER = "\n\n**Final Answer**\n"
MAX_CONTROLLER_STARTER_CHARS = 200

# SGLang caps sampling_seed at int32 max; mirror mining's convention.
MAX_SAMPLING_SEED = 2_147_483_647
DEFAULT_MAX_REASONING_TRACE_TOKENS = 32_768

# Per-turn channel offsets for deterministic sampling.
_CHANNEL_CONTROLLER = 0
_CHANNEL_REASONER = 1
_CHANNEL_FORCED_CONCLUDE = 2
_CHANNEL_FINAL_ANSWER = 3


def _derive_seed(base_seed: int | None, turn_idx: int, channel: int) -> int | None:
    """Derive a per-turn, per-channel sampling seed from a base seed."""
    if base_seed is None:
        return None
    try:
        base = int(base_seed)
    except (TypeError, ValueError):
        return None
    if base < 0:
        return None
    return (base + turn_idx * 4 + channel) % MAX_SAMPLING_SEED


ACTION_PATTERN = re.compile(r"Action\s*:\s*([A-Za-z_]+)", flags=re.IGNORECASE)
STARTER_PATTERN = re.compile(r"Starter\s*:\s*(.+?)(?:\n|$)", flags=re.IGNORECASE)
QUESTION_PATTERN = re.compile(
    r"Question:\s*\n(?P<question>.*?)\n\s*Budget Remaining:\s*\d+%",
    flags=re.DOTALL,
)
BUDGET_REMAINING_PATTERN = re.compile(
    r"Budget Remaining:\s*(?P<pct>-?\d+)%",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_label(label: Any, default_budget: int) -> dict:
    """Parse label into {answer, budget}."""
    answer = ""
    budget = default_budget

    if isinstance(label, dict):
        answer = str(label.get("answer", "")).strip()
        try:
            budget = int(label.get("budget", budget))
        except Exception:
            budget = default_budget
        return {"answer": answer, "budget": max(1, budget)}

    if isinstance(label, str):
        text = label.strip()
        try:
            maybe_json = json.loads(text)
        except Exception:
            maybe_json = None
        if isinstance(maybe_json, dict):
            answer = str(maybe_json.get("answer", "")).strip()
            try:
                budget = int(maybe_json.get("budget", budget))
            except Exception:
                budget = default_budget
            return {"answer": answer, "budget": max(1, budget)}
        answer = text
        return {"answer": answer, "budget": max(1, budget)}

    if label is not None:
        answer = str(label).strip()
    return {"answer": answer, "budget": max(1, budget)}


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _resolve_budget_state(_label_budget: int, metadata: dict[str, Any]) -> tuple[float, float]:
    """Return (initial_remaining_budget, reference_total_budget).

    Full-prompt rollouts start from the question turn. The prompt-visible budget
    must agree with this explicit metadata; do not infer missing budget state
    from any other fields.
    """
    reference_total = _as_float(metadata.get("original_total_budget"))
    if reference_total is None or reference_total <= 0:
        raise ValueError(
            "Full-prompt rollout requires metadata['original_total_budget'] > 0; "
            f"got {metadata.get('original_total_budget')!r}"
        )

    initial_remaining = _as_float(metadata.get("initial_budget_remaining"))
    if initial_remaining is None:
        raise ValueError(
            "Full-prompt rollout requires metadata['initial_budget_remaining']; "
            f"got {metadata.get('initial_budget_remaining')!r}"
        )

    if abs(initial_remaining - reference_total) > 1e-6:
        raise ValueError(
            "Full-prompt rollout must start with the full sampled budget: "
            f"initial_budget_remaining={initial_remaining}, "
            f"original_total_budget={reference_total}"
        )

    return initial_remaining, reference_total


def _validate_start_budget_state(
    *,
    prompt_str: str,
    metadata: dict[str, Any],
    initial_budget_remaining: float,
    reference_total_budget: float,
) -> None:
    """Fail fast if saved prompt budget text disagrees with start metadata."""
    expected_pct = max(
        0,
        int(100 * float(initial_budget_remaining) / float(reference_total_budget)),
    )

    metadata_pct = _as_float(metadata.get("initial_budget_pct"))
    if metadata_pct is None:
        raise ValueError(
            "Full-prompt rollout requires metadata['initial_budget_pct']; "
            f"got {metadata.get('initial_budget_pct')!r}"
        )
    if abs(metadata_pct - expected_pct) > 1.0:
        raise ValueError(
            "Full-prompt budget metadata is inconsistent: "
            f"initial_budget_pct={metadata_pct}, expected={expected_pct}, "
            f"initial_remaining={initial_budget_remaining}, "
            f"original_total={reference_total_budget}"
        )

    prompt_pcts = [int(m.group("pct")) for m in BUDGET_REMAINING_PATTERN.finditer(prompt_str)]
    if not prompt_pcts:
        raise ValueError("Full-prompt rollout prompt has no Budget Remaining field")
    prompt_pct = prompt_pcts[-1]
    if abs(prompt_pct - expected_pct) > 1:
        raise ValueError(
            "Full-prompt budget text is inconsistent with metadata: "
            f"last_prompt_budget_pct={prompt_pct}, expected={expected_pct}, "
            f"initial_remaining={initial_budget_remaining}, "
            f"original_total={reference_total_budget}"
        )


def _extract_question(prompt_str: str) -> str:
    """Extract the original math question from the rendered controller prompt."""
    match = QUESTION_PATTERN.search(prompt_str)
    if match:
        return match.group("question").strip()

    user_blocks = re.findall(
        r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt_str, flags=re.DOTALL
    )
    if user_blocks:
        first_user = user_blocks[0]
        if "Question:\n" in first_user and "\n\nBudget Remaining:" in first_user:
            return (
                first_user.split("Question:\n", 1)[1]
                .split("\n\nBudget Remaining:", 1)[0]
                .strip()
            )
        return first_user.strip()

    return prompt_str.strip()


def _starter_nonascii_fraction(starter: str) -> float:
    if not starter:
        return 0.0
    return sum(1 for ch in starter if ord(ch) > 127) / len(starter)


def _validate_controller_starter(starter: str) -> tuple[bool, str]:
    if not starter:
        return False, "missing_starter"
    if starter.strip().lower() == "none":
        return False, "none_starter"
    if "\n" in starter or "\r" in starter:
        return False, "multiline_starter"
    if len(starter) > MAX_CONTROLLER_STARTER_CHARS:
        return False, "starter_too_long"
    if "\ufffd" in starter or _starter_nonascii_fraction(starter) > 0.20:
        return False, "starter_garbled"
    return True, ""


def _parse_action_output(action_text: str) -> tuple[str, str, bool, str]:
    action_name = None
    starter = ""
    invalid_reasons: list[str] = []

    action_match = ACTION_PATTERN.search(action_text or "")
    if action_match:
        candidate = action_match.group(1).strip().lower()
        if candidate in ALLOWED_ACTIONS:
            action_name = candidate
        else:
            invalid_reasons.append("unknown_action")
    else:
        invalid_reasons.append("missing_action")

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
    else:
        invalid_reasons.append("missing_starter")

    starter_is_valid, starter_invalid_reason = _validate_controller_starter(starter)
    if not starter_is_valid:
        invalid_reasons.append(starter_invalid_reason)

    if action_name is None:
        return "execute", starter or "None", False, ",".join(invalid_reasons)
    if invalid_reasons:
        return action_name, starter or "None", False, ",".join(invalid_reasons)

    return action_name, starter, True, ""


def _prompt_budget_group_key(sample: Sample) -> tuple[Any, ...]:
    metadata = sample.metadata or {}
    label = sample.label if isinstance(sample.label, dict) else {}
    prompt = sample.prompt
    if not isinstance(prompt, str):
        prompt = json.dumps(prompt, sort_keys=True, ensure_ascii=True)
    return (
        prompt,
        label.get("answer"),
        label.get("budget"),
        metadata.get("source_example_id"),
        metadata.get("target_final_budget_pct"),
        metadata.get("original_total_budget"),
        metadata.get("initial_budget_remaining"),
        metadata.get("initial_budget_pct"),
    )


def _validate_group_prompt_budget_consistency(group: list[Sample]) -> None:
    """Ensure one GRPO group shares the same prompt and sampled budget."""
    if len(group) <= 1:
        return
    expected = _prompt_budget_group_key(group[0])
    for idx, sample in enumerate(group[1:], start=1):
        actual = _prompt_budget_group_key(sample)
        if actual != expected:
            raise RuntimeError(
                "GRPO group contains mixed prompts or sampled budgets; "
                f"first_key={expected!r}, sample_{idx}_key={actual!r}"
            )


def _build_reasoner_prompt(question: str, reasoning_trace: str, starter: str, tokenizer) -> str:
    """Build the reasoner prompt using the model's chat template."""
    messages = [{"role": "user", "content": question}]
    base_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    base_prompt = base_prompt.removesuffix("<think>\n").removesuffix("<think>")
    if reasoning_trace:
        return base_prompt + f"<think>\n{reasoning_trace}{starter}"
    return base_prompt + f"<think>\n{starter}"


def _format_env_feedback_ids(tokenizer, step_text: str, budget_pct: int) -> list[int]:
    """Build controller-side user-turn token IDs for reasoner feedback."""
    step_for_prompt = step_text.strip() if step_text.strip() else "<no step content>"
    content = f"Reasoner's Step:\n{step_for_prompt}\n\nBudget Remaining: {budget_pct}%"
    messages = [{"role": "user", "content": content}]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    )
    bos_id = tokenizer.bos_token_id
    if bos_id is not None and ids and ids[0] == bos_id:
        ids = ids[1:]
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    return newline_ids + ids


def _max_reasoning_trace_tokens(args: Any) -> int:
    value = getattr(args, "max_reasoning_trace_tokens", DEFAULT_MAX_REASONING_TRACE_TOKENS)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return DEFAULT_MAX_REASONING_TRACE_TOKENS


def _positive_int_attr(args: Any, name: str) -> int | None:
    value = getattr(args, name, None)
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _response_token_limit(args: Any, prompt_len: int) -> int | None:
    """Return a per-sample response cap.

    Prefer a total train-sequence cap so short prompts can use more
    continuation room while long prompts still fit the training token budget.
    ``rollout_max_response_len`` remains an optional extra hard response cap.
    """
    caps: list[int] = []

    total_cap = _positive_int_attr(args, "max_rollout_total_tokens")
    if total_cap is not None:
        caps.append(max(0, total_cap - int(prompt_len)))

    response_cap = _positive_int_attr(args, "rollout_max_response_len")
    if response_cap is not None:
        caps.append(response_cap)

    return min(caps) if caps else None


def _render_prompt(sample: Sample, tokenizer) -> str:
    prompt_str = sample.prompt if isinstance(sample.prompt, str) else ""
    if not prompt_str and isinstance(sample.prompt, list):
        prompt_str = tokenizer.apply_chat_template(
            sample.prompt, tokenize=False, add_generation_prompt=True,
        )
    return prompt_str


def _route_headers(args: Any, sample: Sample, role: str) -> dict | None:
    if getattr(args, "router_policy", None) not in {"consistent_hashing", "manual"}:
        return None
    route_id = sample.session_id or f"{sample.group_index}:{sample.index}"
    return {"X-SMG-Routing-Key": f"acts:{role}:{route_id}"}


def _append_response_tokens(
    *,
    sample: Sample,
    response_tokens: list[int],
    token_ids: list[int],
    loss_value: int,
    rollout_log_probs: list[float] | None = None,
) -> None:
    if not token_ids:
        return
    if rollout_log_probs is None:
        rollout_log_probs = [0.0] * len(token_ids)
    if len(rollout_log_probs) != len(token_ids):
        raise ValueError(
            f"rollout_log_probs length {len(rollout_log_probs)} "
            f"!= token length {len(token_ids)}"
        )
    sample.tokens.extend(token_ids)
    response_tokens.extend(token_ids)
    sample.loss_mask.extend([int(loss_value)] * len(token_ids))
    sample.rollout_log_probs.extend(rollout_log_probs)
    sample.response_length = len(response_tokens)


def _fits_response_limit(current_len: int, add_len: int, limit: int | None) -> bool:
    if limit is None:
        return True
    return current_len + add_len <= limit


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


async def _call_controller(
    *,
    controller_url: str,
    input_ids: list[int],
    sampling_params: dict,
    tokenizer,
    ctrl_eos_id: int | None,
    headers: dict | None = None,
) -> tuple[str, list[int], list[float]]:
    """Call the controller and return (text, token_ids, token_logprobs)."""
    output = await _tracked_post(
        controller_url,
        {
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
        },
        kind="ctrl",
        headers=headers,
    )
    text = output["text"]
    meta = output["meta_info"]
    if "output_token_logprobs" in meta:
        token_ids = [item[1] for item in meta["output_token_logprobs"]]
        token_logprobs = [item[0] for item in meta["output_token_logprobs"]]
    else:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if ctrl_eos_id is not None and (not token_ids or token_ids[-1] != ctrl_eos_id):
            token_ids = token_ids + [ctrl_eos_id]
        token_logprobs = [0.0] * len(token_ids)
    return text, token_ids, token_logprobs


async def _call_reasoner(
    url: str, prompt: str, sampling_params: dict, headers: dict | None = None,
) -> tuple[str, str]:
    """Call the reasoner SGLang engine with a text prompt. Returns (text, finish_type)."""
    payload = {
        "text": prompt,
        "sampling_params": sampling_params,
    }
    output = await _tracked_post(url, payload, kind="reasoner", headers=headers)
    text = output["text"]
    meta = output["meta_info"]
    finish_type = meta["finish_reason"]["type"]
    return text, finish_type


async def _generate_forced_conclude(
    reasoner_url: str,
    question: str,
    reasoning_trace: str,
    args: Any,
    tokenizer,
    *,
    sampling_seed: int | None = None,
    headers: dict | None = None,
) -> str:
    prompt = _build_reasoner_prompt(question, reasoning_trace, FORCED_CONCLUDE_STARTER, tokenizer)
    sampling_params = {
        "temperature": args.reasoner_temperature,
        "top_p": args.reasoner_top_p,
        "max_new_tokens": args.reasoner_forced_conclude_max_tokens,
        "stop": ["</think>"],
        "no_stop_trim": True,
    }
    if sampling_seed is not None and sampling_seed >= 0:
        sampling_params["sampling_seed"] = int(sampling_seed)
    text, finish_type = await _call_reasoner(
        reasoner_url, prompt, sampling_params, headers=headers,
    )

    forced_text = FORCED_CONCLUDE_STARTER + text
    if finish_type == "stop" and not forced_text.rstrip().endswith("</think>"):
        forced_text += "</think>"
    if not forced_text.rstrip().endswith("</think>"):
        forced_text = forced_text.rstrip() + "\n</think>"
    return forced_text


async def _generate_final_answer(
    reasoner_url: str,
    question: str,
    reasoning_trace: str,
    args: Any,
    tokenizer,
    *,
    sampling_seed: int | None = None,
    headers: dict | None = None,
) -> str:
    """Generate the post-</think> answer channel for reward grading."""
    trace = reasoning_trace.rstrip()
    if not trace.endswith("</think>"):
        trace += "\n</think>"

    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    prompt = prompt.removesuffix("<think>\n").removesuffix("<think>")
    prompt += f"<think>\n{trace}\n\n"

    sampling_params = {
        "temperature": getattr(args, "reasoner_temperature", 0.0),
        "top_p": getattr(args, "reasoner_top_p", 1.0),
        "max_new_tokens": getattr(args, "reasoner_final_answer_max_tokens", 2048),
    }
    if sampling_seed is not None and sampling_seed >= 0:
        sampling_params["sampling_seed"] = int(sampling_seed)
    payload = {"text": prompt, "sampling_params": sampling_params}
    output = await _tracked_post(
        reasoner_url, payload, kind="reasoner", headers=headers,
    )
    return output["text"]


# ---------------------------------------------------------------------------
# Full continuation rollout
# ---------------------------------------------------------------------------


async def _generate_full_continuation(
    args: Any,
    sample: Sample,
    sampling_params: dict,
    evaluation: bool,
    *,
    tokenizer,
    budget_tokenizer,
    controller_url: str,
    reasoner_url: str,
) -> Sample:
    metadata = sample.metadata or {}
    default_budget = metadata.get("default_budget") or getattr(args, "default_budget", 3072)
    label = _parse_label(sample.label, default_budget)
    ground_truth = label["answer"]
    initial_budget_remaining, reference_total_budget = _resolve_budget_state(
        label["budget"], metadata,
    )

    prompt_str = _render_prompt(sample, tokenizer)
    _validate_start_budget_state(
        prompt_str=prompt_str,
        metadata=metadata,
        initial_budget_remaining=initial_budget_remaining,
        reference_total_budget=reference_total_budget,
    )
    question = _extract_question(prompt_str)
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    if not sample.tokens:
        sample.tokens = list(prompt_ids)
    prompt_token_len = len(sample.tokens)

    response_tokens: list[int] = []
    sample.loss_mask = []
    sample.rollout_log_probs = []
    sample.metadata = metadata

    ctrl_headers = _route_headers(args, sample, "ctrl")
    reasoner_headers = _route_headers(args, sample, "reasoner")

    max_turns = int(getattr(args, "max_turns", 500))
    max_reasoning_trace_tokens = _max_reasoning_trace_tokens(args)
    response_limit = _response_token_limit(args, prompt_token_len)
    reasoner_stop = getattr(args, "reasoner_stop_sequences", DEFAULT_STOP_SEQUENCES)
    ctrl_eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ctrl_stop_ids = [ctrl_eos_id] if ctrl_eos_id is not None else []
    base_sampling_seed = sampling_params.get("sampling_seed")
    initial_reasoning_trace = str(metadata.get("reasoning_trace_so_far") or "")
    local_reasoning_trace = initial_reasoning_trace
    local_real_budget_remaining = float(initial_budget_remaining)
    local_reasoning_trace_tokens = 0
    local_step_idx = 0
    local_forced_conclude_count = 0
    local_done = False
    trace_cap_reached = False
    response_cap_reached = False
    controller_actions: list[str] = []
    controller_starters: list[str] = []
    controller_token_counts: list[int] = []
    latest_action = ""
    latest_starter = ""
    invalid_controller_output = False
    invalid_controller_reason = ""
    invalid_controller_turn_idx: int | None = None

    for turn_idx in range(max_turns):
        if response_limit is not None and len(response_tokens) >= response_limit:
            response_cap_reached = True
            break

        ctrl_sampling_params = {
            "temperature": sampling_params.get("temperature", 1.0),
            "top_p": sampling_params.get("top_p", 1.0),
            "max_new_tokens": 64,
            "stop_token_ids": ctrl_stop_ids,
        }
        ctrl_seed = _derive_seed(base_sampling_seed, turn_idx, _CHANNEL_CONTROLLER)
        if ctrl_seed is not None:
            ctrl_sampling_params["sampling_seed"] = ctrl_seed

        ctrl_text, ctrl_tokens, ctrl_logprobs = await _call_controller(
            controller_url=controller_url,
            input_ids=sample.tokens,
            sampling_params=ctrl_sampling_params,
            tokenizer=tokenizer,
            ctrl_eos_id=ctrl_eos_id,
            headers=ctrl_headers,
        )
        if not _fits_response_limit(len(response_tokens), len(ctrl_tokens), response_limit):
            response_cap_reached = True
            break

        _append_response_tokens(
            sample=sample,
            response_tokens=response_tokens,
            token_ids=ctrl_tokens,
            loss_value=1,
            rollout_log_probs=ctrl_logprobs,
        )

        action, starter, controller_output_valid, controller_invalid_reason = _parse_action_output(ctrl_text)
        latest_action = action
        latest_starter = starter
        controller_actions.append(action)
        controller_starters.append(starter)
        controller_token_counts.append(len(ctrl_tokens))

        starter_for_reasoner = ""
        if starter and starter.strip().lower() != "none":
            starter_for_reasoner = starter

        reasoner_prompt = _build_reasoner_prompt(
            question, local_reasoning_trace, starter_for_reasoner, budget_tokenizer,
        )
        reasoner_sampling_params = {
            "temperature": getattr(args, "reasoner_temperature", 0.0),
            "top_p": getattr(args, "reasoner_top_p", 1.0),
            "max_new_tokens": getattr(args, "reasoner_max_tokens_per_step", 512),
            "stop": list(reasoner_stop),
            "no_stop_trim": True,
        }
        reasoner_seed = _derive_seed(base_sampling_seed, turn_idx, _CHANNEL_REASONER)
        if reasoner_seed is not None:
            reasoner_sampling_params["sampling_seed"] = reasoner_seed

        reasoner_text, finish_type = await _call_reasoner(
            reasoner_url,
            reasoner_prompt,
            reasoner_sampling_params,
            headers=reasoner_headers,
        )

        step_text = starter_for_reasoner + reasoner_text
        step_done = False
        if finish_type == "stop":
            if "</think>" in reasoner_text or step_text.rstrip().endswith("</think>"):
                step_done = True
                if not step_text.rstrip().endswith("</think>"):
                    step_text += "</think>"
        if "</think>" in step_text:
            step_done = True

        local_reasoning_trace += step_text
        budget_text = step_text.replace("</think>", "").strip()
        tokens_used = len(budget_tokenizer.encode(budget_text, add_special_tokens=False))
        local_real_budget_remaining -= float(tokens_used)
        local_reasoning_trace_tokens += tokens_used
        local_step_idx += 1

        if (
            max_reasoning_trace_tokens > 0
            and local_reasoning_trace_tokens >= max_reasoning_trace_tokens
            and not step_done
        ):
            trace_cap_reached = True
            break

        if action == "conclude" and not step_done:
            forced_seed = _derive_seed(
                base_sampling_seed, turn_idx, _CHANNEL_FORCED_CONCLUDE,
            )
            forced_text = await _generate_forced_conclude(
                reasoner_url,
                question,
                local_reasoning_trace,
                args,
                budget_tokenizer,
                sampling_seed=forced_seed,
                headers=reasoner_headers,
            )
            local_reasoning_trace += forced_text
            step_text += forced_text
            step_done = True
            local_forced_conclude_count += 1
            forced_budget_text = forced_text.replace("</think>", "").strip()
            forced_tokens = len(
                budget_tokenizer.encode(forced_budget_text, add_special_tokens=False)
            )
            local_real_budget_remaining -= float(forced_tokens)
            local_reasoning_trace_tokens += forced_tokens

        local_done = step_done or action == "conclude"

        if local_done:
            break

        # Use the sampled total budget as denominator. Do not normalize by the
        # current remaining budget; that would change the controller input scale.
        budget_pct = max(
            0,
            int(100 * local_real_budget_remaining / float(reference_total_budget)),
        )
        env_token_ids = _format_env_feedback_ids(tokenizer, step_text, budget_pct)
        if not _fits_response_limit(len(response_tokens), len(env_token_ids), response_limit):
            response_cap_reached = True
            break

        _append_response_tokens(
            sample=sample,
            response_tokens=response_tokens,
            token_ids=env_token_ids,
            loss_value=0,
            rollout_log_probs=[0.0] * len(env_token_ids),
        )

    answer_text = ""
    generate_answer_text = getattr(args, "generate_answer_text_for_training", True)
    if local_done and local_reasoning_trace.strip() and (evaluation or generate_answer_text):
        final_seed = _derive_seed(
            base_sampling_seed, max_turns, _CHANNEL_FINAL_ANSWER,
        )
        answer_text = await _generate_final_answer(
            reasoner_url,
            question,
            local_reasoning_trace,
            args,
            budget_tokenizer,
            sampling_seed=final_seed,
            headers=reasoner_headers,
        )

    sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    sample.status = Sample.Status.COMPLETED if local_done else Sample.Status.TRUNCATED

    sample.metadata["reasoning_trace"] = local_reasoning_trace
    sample.metadata["answer_text"] = answer_text
    sample.metadata["total_budget"] = reference_total_budget
    sample.metadata["initial_budget_remaining"] = initial_budget_remaining
    sample.metadata["ground_truth"] = ground_truth
    sample.metadata["real_budget_remaining"] = local_real_budget_remaining
    sample.metadata["num_steps"] = local_step_idx
    sample.metadata["forced_conclude_count"] = local_forced_conclude_count
    sample.metadata["trace_cap_reached"] = trace_cap_reached
    sample.metadata["response_cap_reached"] = response_cap_reached
    sample.metadata["invalid_controller_output"] = invalid_controller_output
    sample.metadata["invalid_controller_reason"] = invalid_controller_reason
    sample.metadata["invalid_controller_turn_idx"] = invalid_controller_turn_idx
    sample.metadata["rollout_completed"] = local_done
    sample.metadata["reasoning_trace_tokens"] = local_reasoning_trace_tokens
    sample.metadata["max_reasoning_trace_tokens"] = max_reasoning_trace_tokens
    sample.metadata["rollout_max_response_len"] = response_limit
    sample.metadata["max_rollout_total_tokens"] = _positive_int_attr(
        args, "max_rollout_total_tokens"
    )
    sample.metadata["controller_turn_count"] = len(controller_actions)
    sample.metadata["controller_actions"] = controller_actions
    sample.metadata["controller_starters"] = controller_starters
    sample.metadata["controller_token_counts"] = controller_token_counts
    sample.metadata["first_chosen_action"] = controller_actions[0] if controller_actions else ""
    sample.metadata["first_chosen_starter"] = controller_starters[0] if controller_starters else ""
    sample.metadata["last_chosen_action"] = latest_action
    sample.metadata["last_chosen_starter"] = latest_starter
    sample.metadata["trainable_controller_tokens"] = int(sum(sample.loss_mask))
    sample.metadata["continuation_objective"] = "full_controller_trajectory"
    sample.metadata["budget_reference_total_source"] = "metadata.original_total_budget"
    sample.metadata["budget_initial_remaining_source"] = "metadata.initial_budget_remaining"

    if len(sample.loss_mask) != sample.response_length:
        raise RuntimeError(
            f"loss mask length {len(sample.loss_mask)} != response length "
            f"{sample.response_length}"
        )
    if sample.rollout_log_probs is not None and len(sample.rollout_log_probs) != sample.response_length:
        raise RuntimeError(
            f"rollout_log_probs length {len(sample.rollout_log_probs)} != response length "
            f"{sample.response_length}"
        )
    total_cap = _positive_int_attr(args, "max_rollout_total_tokens")
    if total_cap is not None and len(sample.tokens) > total_cap:
        raise RuntimeError(
            f"sample total length {len(sample.tokens)} exceeds max_rollout_total_tokens "
            f"{total_cap}; prompt_len={prompt_token_len}, response_len={sample.response_length}"
        )

    return sample


# ---------------------------------------------------------------------------
# Main generate functions
# ---------------------------------------------------------------------------


async def generate_group(
    args: Any, group: list[Sample], sampling_params: dict, evaluation: bool = False
) -> list[Sample]:
    """Entry point wired into SLIME. Generate one GRPO group as independent
    full controller-steered continuations."""
    if not group:
        return group

    for _ in group:
        await _rollout_stats.sample_start()
    try:
        state = GenerateState(args)
        async with state.semaphore:
            return await _generate_group_impl(args, group, sampling_params, evaluation, state)
    finally:
        for _ in group:
            await _rollout_stats.sample_done()


async def _generate_group_impl(
    args: Any,
    group: list[Sample],
    sampling_params: dict,
    evaluation: bool,
    state: GenerateState,
) -> list[Sample]:
    tokenizer = state.tokenizer
    reasoner_model_path = getattr(args, "reasoner_model_path", None)
    budget_tokenizer = (
        _get_reasoner_tokenizer(reasoner_model_path)
        if reasoner_model_path
        else tokenizer
    )

    controller_url = get_model_url(args, "controller", "/generate")
    reasoner_url = get_model_url(args, "reasoner", "/generate")
    _validate_group_prompt_budget_consistency(group)

    if not getattr(generate_group, "_config_logged", False):
        generate_group._config_logged = True
        logger.info(
            f"[acts_rollout] objective=full_controller_trajectory "
            f"controller_url={controller_url} "
            f"reasoner_url={reasoner_url} "
            f"reasoner_model_path={reasoner_model_path}"
        )

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            current_sampling_params["sampling_seed"] = state.group_sampling_seeds[idx]
        tasks.append(
            _generate_full_continuation(
                args,
                sample,
                current_sampling_params,
                evaluation,
                tokenizer=tokenizer,
                budget_tokenizer=budget_tokenizer,
                controller_url=controller_url,
                reasoner_url=reasoner_url,
            )
        )

    return await asyncio.gather(*tasks)
