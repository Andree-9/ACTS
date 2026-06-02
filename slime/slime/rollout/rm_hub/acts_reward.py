"""
Custom reward function for ACTS controller-reasoner RL training.

Wired via: --custom-rm-path slime.rollout.rm_hub.acts_reward.custom_rm

Correctness-dominant asymmetric budget reward:
  correct, within budget / small overrun -> 1.0
  correct, large overrun                 -> 1.0 - mild over-budget penalty
  incorrect, over budget / cap hit       -> mild over-budget penalty
  incorrect, near budget                 -> 0.0
  incorrect, budget left beyond grace    -> underuse penalty
"""

from __future__ import annotations

import json
import re
from typing import Any

import torch

from evaluation.scoring import extract_answer_candidate, score_response, score_responses
from slime.utils.types import Sample

DEFAULT_OVER_BUDGET_GRACE_FRAC = 0.10
DEFAULT_BUDGET_SAVED_GRACE_FRAC = 0.10
DEFAULT_CORRECT_OVER_BUDGET_ALPHA = 0.25
DEFAULT_WRONG_UNDER_BUDGET_ALPHA = 0.75
FINAL_STARTER_PATTERN = re.compile(r"(?:^|\*)\s*final\b", flags=re.IGNORECASE)
# ---------------------------------------------------------------------------
# Label parsing (same as acts_rollout._parse_label)
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


def _build_response_for_grading(sample: Sample) -> str:
    metadata = sample.metadata or {}
    if metadata.get("rollout_completed") is False:
        return ""
    reasoning_trace = metadata.get("reasoning_trace") or sample.response or ""
    answer_text = metadata.get("answer_text") or ""

    return answer_text or reasoning_trace


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _budget_reward_config(args: Any) -> dict[str, float]:
    """Read asymmetric budget reward knobs from the custom config."""
    over_budget_grace_frac = float(
        getattr(args, "over_budget_grace_frac", DEFAULT_OVER_BUDGET_GRACE_FRAC)
    )
    budget_saved_grace_frac = float(
        getattr(args, "budget_saved_grace_frac", DEFAULT_BUDGET_SAVED_GRACE_FRAC)
    )
    correct_over_budget_alpha = float(
        getattr(args, "correct_over_budget_alpha", DEFAULT_CORRECT_OVER_BUDGET_ALPHA)
    )
    wrong_under_budget_alpha = float(
        getattr(args, "wrong_under_budget_alpha", DEFAULT_WRONG_UNDER_BUDGET_ALPHA)
    )
    return {
        "over_budget_grace_frac": _clamp(over_budget_grace_frac, 0.0, 0.999),
        "budget_saved_grace_frac": _clamp(budget_saved_grace_frac, 0.0, 0.999),
        "correct_over_budget_alpha": max(0.0, correct_over_budget_alpha),
        "wrong_under_budget_alpha": max(0.0, wrong_under_budget_alpha),
    }


def _resolve_total_budget(metadata: dict[str, Any]) -> float | None:
    for key in ("total_budget", "original_total_budget", "initial_budget_remaining"):
        total_budget = _safe_float(metadata.get(key))
        if total_budget is not None and total_budget > 0:
            return total_budget
    return None


def _last_controller_starter_requests_final(metadata: dict[str, Any]) -> bool:
    """Whether the controller's last starter explicitly asks for final answer."""
    starter = metadata.get("last_chosen_starter")
    if not isinstance(starter, str):
        starters = metadata.get("controller_starters")
        if isinstance(starters, list) and starters:
            starter = starters[-1]
    if not isinstance(starter, str):
        return False
    return bool(FINAL_STARTER_PATTERN.search(starter.strip()))


def _budget_reward_components(
    metadata: dict[str, Any],
    *,
    correctness: float,
    remaining_budget: float | None,
    reward_config: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    """Compute the scalar reward plus loggable budget-shaping components."""
    c = _clamp(correctness, 0.0, 1.0)
    total_budget = _resolve_total_budget(metadata)
    if remaining_budget is None or total_budget is None:
        return c, {
            "budget_remaining_frac": None,
            "budget_over_frac": None,
            "budget_effective_over_frac": None,
            "budget_saved_frac": None,
            "correct_over_budget_penalty": 0.0,
            "wrong_under_budget_penalty": 0.0,
            "wrong_over_budget_penalty": 0.0,
            "budget_penalty_applied": 0.0,
            "is_over_budget": bool(metadata.get("trace_cap_reached")),
            "is_within_budget": None,
            "is_budget_underused": None,
        }

    remaining_frac = remaining_budget / total_budget
    over_frac = max(0.0, -remaining_frac)
    saved_frac = max(0.0, remaining_frac)

    over_grace = reward_config["over_budget_grace_frac"]
    saved_grace = reward_config["budget_saved_grace_frac"]
    correct_alpha = reward_config["correct_over_budget_alpha"]
    wrong_alpha = reward_config["wrong_under_budget_alpha"]

    effective_over_frac = over_frac
    if bool(metadata.get("trace_cap_reached")):
        effective_over_frac = max(effective_over_frac, over_grace + 1.0)

    correct_over_excess = max(0.0, effective_over_frac - over_grace)
    correct_over_penalty = correct_alpha * min(correct_over_excess, 1.0)
    correct_reward = 1.0 - correct_over_penalty

    wrong_saved_penalty = 0.0
    if _last_controller_starter_requests_final(metadata):
        wrong_saved_excess = max(0.0, saved_frac - saved_grace)
        wrong_saved_penalty = wrong_alpha * min(
            wrong_saved_excess / max(1e-9, 1.0 - saved_grace),
            1.0,
        )
    wrong_over_budget_penalty = correct_over_penalty
    wrong_reward = -(wrong_saved_penalty + wrong_over_budget_penalty)

    reward = c * correct_reward + (1.0 - c) * wrong_reward
    budget_penalty = c * correct_over_penalty + (1.0 - c) * (
        wrong_saved_penalty + wrong_over_budget_penalty
    )
    is_over_budget = bool(metadata.get("trace_cap_reached")) or over_frac > over_grace

    return reward, {
        "budget_remaining_frac": remaining_frac,
        "budget_over_frac": over_frac,
        "budget_effective_over_frac": effective_over_frac,
        "budget_saved_frac": saved_frac,
        "correct_over_budget_penalty": correct_over_penalty,
        "wrong_under_budget_penalty": wrong_saved_penalty,
        "wrong_over_budget_penalty": wrong_over_budget_penalty,
        "budget_penalty_applied": budget_penalty,
        "is_over_budget": is_over_budget,
        "is_within_budget": not is_over_budget,
        "is_budget_underused": saved_frac > saved_grace,
    }


def post_process_rewards(args: Any, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """ACTS reward post-processing for controller GRPO.

    Keeps the scorer-produced rewards unchanged, then applies SLIME's usual
    GRPO group mean-centering and optional std normalization.
    """
    raw_rewards = [float(sample.get_reward_value(args)) for sample in samples]
    rewards = list(raw_rewards)

    if (
        getattr(args, "advantage_estimator", None)
        in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and getattr(args, "rewards_normalization", True)
    ):
        rewards_tensor = torch.tensor(rewards, dtype=torch.float)
        expected_size = int(getattr(args, "n_samples_per_prompt", 1)) * int(
            getattr(args, "rollout_batch_size", 1)
        )
        if rewards_tensor.shape[-1] == expected_size:
            rewards_tensor = rewards_tensor.reshape(-1, int(getattr(args, "n_samples_per_prompt", 1)))
        else:
            rewards_tensor = rewards_tensor.view(-1, rewards_tensor.shape[-1])

        rewards_tensor = rewards_tensor - rewards_tensor.mean(dim=-1, keepdim=True)

        if (
            getattr(args, "advantage_estimator", None) in ["grpo", "gspo"]
            and getattr(args, "grpo_std_normalization", True)
        ):
            std = rewards_tensor.std(dim=-1, keepdim=True)
            rewards_tensor = rewards_tensor / (std + 1e-6)

        return raw_rewards, rewards_tensor.flatten().tolist()

    return raw_rewards, rewards


def _finalize_reward_metadata(
    metadata: dict[str, Any],
    *,
    full_output: str,
    is_correct: float,
    reward_config: dict[str, float],
) -> float:
    """Finalize reward bookkeeping for a single continuation."""
    remaining_budget = _safe_float(metadata.get("real_budget_remaining"))
    correctness = float(is_correct)
    reward, budget_components = _budget_reward_components(
        metadata,
        correctness=correctness,
        remaining_budget=remaining_budget,
        reward_config=reward_config,
    )
    metadata["is_correct"] = correctness
    metadata["extracted_answer"] = extract_answer_candidate(full_output)
    metadata.update(reward_config)
    metadata.update(budget_components)

    return reward


# ---------------------------------------------------------------------------
# Custom reward function
# ---------------------------------------------------------------------------


def _score_single(args: Any, sample: Sample) -> float:
    """Compute reward for a single sample."""
    metadata = sample.metadata or {}
    default_budget = metadata.get("default_budget") or getattr(args, "default_budget", 3072)
    reward_config = _budget_reward_config(args)

    ground_truth = metadata.get("ground_truth", "")
    if not ground_truth:
        ground_truth = _parse_label(sample.label, default_budget)["answer"]

    grader_backend = getattr(args, "grader_backend", "math_verify")
    timeout_score = getattr(args, "grader_timeout_score", 0.0)

    full_output = _build_response_for_grading(sample)
    is_correct = score_response(
        full_output,
        ground_truth,
        backend=grader_backend,
        timeout_score=timeout_score,
    )

    return _finalize_reward_metadata(
        metadata,
        full_output=full_output,
        is_correct=float(bool(is_correct)),
        reward_config=reward_config,
    )


async def custom_rm(
    args: Any, sample_or_samples: Sample | list[Sample], **kwargs
) -> float | list[float]:
    """
    Reward for ACTS controller RL.

    Called by SLIME's async_rm() for a single sample, or by batched_async_rm()
    with a list of samples. Must handle both signatures.
    """
    if isinstance(sample_or_samples, list):
        samples = sample_or_samples
        metadata_list = [sample.metadata or {} for sample in samples]
        default_budget = getattr(args, "default_budget", 3072)
        reward_config = _budget_reward_config(args)

        # One answer per sample; grade them all in a single batched call.
        ground_truths: list[str] = []
        answers: list[str] = []
        for sample, metadata in zip(samples, metadata_list, strict=False):
            sample_default_budget = metadata.get("default_budget") or default_budget
            ground_truth = metadata.get("ground_truth", "")
            if not ground_truth:
                ground_truth = _parse_label(sample.label, sample_default_budget)["answer"]
            ground_truths.append(ground_truth)
            answers.append(_build_response_for_grading(sample))

        grader_backend = getattr(args, "grader_backend", "math_verify")
        timeout_score = getattr(args, "grader_timeout_score", 0.0)
        is_correct_list = score_responses(
            answers,
            ground_truths,
            backend=grader_backend,
            timeout_score=timeout_score,
        )

        rewards: list[float] = []
        for metadata, full_output, is_correct in zip(
            metadata_list, answers, is_correct_list, strict=False
        ):
            rewards.append(
                _finalize_reward_metadata(
                    metadata,
                    full_output=full_output,
                    is_correct=float(bool(is_correct)),
                    reward_config=reward_config,
                )
            )
        return rewards
    return _score_single(args, sample_or_samples)
