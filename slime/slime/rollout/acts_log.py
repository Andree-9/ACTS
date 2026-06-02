"""
Custom rollout log function for ACTS controller-reasoner RL training.

Wired via: --custom-rollout-log-function-path slime.rollout.acts_log.log_rollout_data

Logs domain-specific metrics alongside SLIME's default rollout metrics:
  - correctness, rewards
  - within-group reward std (GRPO signal strength)
  - first-action mean reward std (diagnostic only)
  - within-group starter diversity (controller exploration health)
  - budget usage, step counts
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

import numpy as np

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step

logger = logging.getLogger(__name__)

_DUMP_DONE = False


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _compute_budget_stats(meta: dict[str, Any]) -> tuple[float | None, float | None]:
    total_budget = _safe_float(meta.get("total_budget"))
    remaining_budget = _safe_float(meta.get("real_budget_remaining"))
    if total_budget is None or remaining_budget is None or total_budget <= 0:
        return None, None
    remaining_pct = max(0.0, min(100.0, 100.0 * remaining_budget / total_budget))
    used_pct = max(0.0, 100.0 - remaining_pct)
    return used_pct, remaining_pct


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return float(np.mean(present))


def _action_key(meta: dict[str, Any]) -> str | None:
    action = str(meta.get("first_chosen_action") or "").strip().lower()
    return action or None


def _action_mean_rewards_for_group(group: list) -> list[float]:
    """Diagnostic first-action reward grouping for one GRPO group."""
    raw_rewards: list[float] = []
    action_to_indices: dict[str, list[int]] = {}

    for idx, sample in enumerate(group):
        reward = _safe_float(sample.reward)
        if reward is None:
            continue
        raw_rewards.append(reward)

        action = _action_key(sample.metadata or {})
        if action is not None:
            action_to_indices.setdefault(action, []).append(len(raw_rewards) - 1)

    action_mean_rewards = list(raw_rewards)
    for indices in action_to_indices.values():
        mean_reward = float(np.mean([raw_rewards[idx] for idx in indices]))
        for idx in indices:
            action_mean_rewards[idx] = mean_reward

    return action_mean_rewards


def _group_signal_flags(group: list) -> tuple[float, float]:
    """Return correctness-signal and budget-only-signal flags for one group."""
    correctness = [
        float(value)
        for sample in group
        if (value := _safe_float((sample.metadata or {}).get("is_correct"))) is not None
    ]
    rewards = [
        float(value)
        for sample in group
        if (value := _safe_float(sample.reward)) is not None
    ]

    correctness_var = len(correctness) >= 2 and float(np.std(correctness)) > 1e-12
    reward_var = len(rewards) >= 2 and float(np.std(rewards)) > 1e-12
    budget_only_var = reward_var and not correctness_var
    return float(correctness_var), float(budget_only_var)


def _resolve_budget_ratios(meta: dict[str, Any]) -> tuple[float | None, float | None]:
    mc_is_over_budget = meta.get("mc_is_over_budget")
    if isinstance(mc_is_over_budget, list) and mc_is_over_budget:
        over_budget_ratio = float(np.mean([1.0 if bool(v) else 0.0 for v in mc_is_over_budget]))
        return 1.0 - over_budget_ratio, over_budget_ratio

    mc_within_budget_ratio = _safe_float(meta.get("mc_within_budget_ratio"))
    if mc_within_budget_ratio is not None:
        within_budget_ratio = max(0.0, min(1.0, mc_within_budget_ratio))
        return within_budget_ratio, 1.0 - within_budget_ratio

    is_within_budget = meta.get("is_within_budget")
    is_over_budget = meta.get("is_over_budget")
    if isinstance(is_within_budget, bool) and isinstance(is_over_budget, bool):
        return float(is_within_budget), float(is_over_budget)

    total_budget = _safe_float(meta.get("total_budget"))
    remaining_budget = _safe_float(meta.get("real_budget_remaining"))
    over_budget_grace_frac = _safe_float(meta.get("over_budget_grace_frac"))
    if remaining_budget is None or total_budget is None or total_budget <= 0:
        return None, None
    if over_budget_grace_frac is None:
        over_budget_grace_frac = 0.10

    resolved_over_budget = (
        bool(meta.get("trace_cap_reached"))
        or remaining_budget / total_budget < -over_budget_grace_frac
    )
    return float(not resolved_over_budget), float(resolved_over_budget)


def _truncate_text(value: Any, limit: int = 20_000) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n...[truncated {len(text) - limit} chars]"


def _format_prompt_for_table(prompt: Any) -> str:
    if isinstance(prompt, list):
        parts = []
        for message in prompt:
            if isinstance(message, dict):
                role = message.get("role", "")
                content = message.get("content", "")
                parts.append(f"### {role}\n{content}")
            else:
                parts.append(str(message))
        return "\n\n".join(parts)
    return str(prompt)


def _reconstruct_groups(args: Any, samples: list, flat_samples: list) -> list[list]:
    """Return a list of GRPO groups.

    ``samples`` may be ``list[list[Sample]]`` (already grouped) or a flat list.
    When flat, groups are reconstructed by chunking on ``n_samples_per_prompt``.
    """
    if samples and isinstance(samples[0], list):
        return list(samples)
    n = getattr(args, "n_samples_per_prompt", 1) or 1
    return [flat_samples[i : i + n] for i in range(0, len(flat_samples), n)]


def _log_sample_group_table(args: Any, groups: list[list], rollout_id: int, step: int) -> None:
    """Log one sampled GRPO group as a W&B table for behavior inspection."""
    if not groups:
        return

    non_empty_groups = [g for g in groups if g]
    if not non_empty_groups:
        return

    rng = random.Random(int(rollout_id))
    group_pos = rng.randrange(len(non_empty_groups))
    group = non_empty_groups[group_pos]

    project_dir = os.environ.get(
        "ACTS_PROJECT_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", ".."),
    )
    sample_log_path = os.path.join(project_dir, "logs", "controller_sample_groups.jsonl")
    try:
        os.makedirs(os.path.dirname(sample_log_path), exist_ok=True)
        prompt_prefix = _truncate_text(_format_prompt_for_table(group[0].prompt), 20_000)
        with open(sample_log_path, "a", encoding="utf-8") as f:
            for sample_pos, sample in enumerate(group):
                meta = sample.metadata or {}
                row = {
                    "rollout_step": step,
                    "rollout_id": rollout_id,
                    "selected_group_idx": group_pos,
                    "sample_pos": sample_pos,
                    "prompt_prefix": prompt_prefix if sample_pos == 0 else "",
                    "controller_response": sample.response or "",
                    "controller_actions": meta.get("controller_actions"),
                    "controller_starters": meta.get("controller_starters"),
                    "controller_turn_count": meta.get("controller_turn_count"),
                    "real_budget_remaining": meta.get("real_budget_remaining"),
                    "answer_text": meta.get("answer_text") or "",
                    "reward": _safe_float(sample.reward),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(
            "[acts_log] Appended %d sampled controller rows to %s",
            len(group),
            sample_log_path,
        )
    except Exception as e:
        logger.warning(f"[acts_log] Failed to write sample group JSONL: {e}")

def _dump_first_batch(args: Any, samples: list, flat_samples: list, rollout_id: int) -> None:
    """Save the first rollout round as grouped summaries for debugging."""
    project_dir = os.environ.get(
        "ACTS_PROJECT_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", ".."),
    )
    dump_path = os.path.join(project_dir, "first_rollout_dump.json")

    groups = _reconstruct_groups(args, samples, flat_samples)[:16]

    dump = {
        "rollout_id": rollout_id,
        "n_groups": len(groups),
        "n_samples_per_prompt": getattr(args, "n_samples_per_prompt", None),
        "rollout_batch_size": getattr(args, "rollout_batch_size", None),
        "groups": [],
    }

    for gi, group in enumerate(groups):
        rewards = []
        correctness = []
        budget_used_pcts = []
        budget_remaining_pcts = []
        num_steps = []
        response_lengths = []
        chosen_actions = []
        chosen_starters = []
        group_data = {"group_idx": gi, "n_samples": len(group), "samples": []}
        for si, sample in enumerate(group):
            meta = sample.metadata or {}
            budget_used_pct, budget_remaining_pct = _compute_budget_stats(meta)
            within_budget_ratio, over_budget_ratio = _resolve_budget_ratios(meta)
            is_correct = float(_safe_float(meta.get("is_correct")) or 0.0)
            reward = sample.reward
            rewards.append(float(reward) if reward is not None else None)
            correctness.append(is_correct)
            budget_used_pcts.append(budget_used_pct)
            budget_remaining_pcts.append(budget_remaining_pct)
            num_steps.append(_safe_float(meta.get("num_steps")))
            response_lengths.append(_safe_float(sample.response_length))
            chosen_actions.append(meta.get("first_chosen_action") or "")
            chosen_starters.append(meta.get("first_chosen_starter") or "")
            group_data["samples"].append({
                "sample_idx": si,
                "source_idx": meta.get("source_idx"),
                "total_budget": meta.get("total_budget"),
                "real_budget_remaining": meta.get("real_budget_remaining"),
                "budget_used_pct": budget_used_pct,
                "budget_remaining_pct": budget_remaining_pct,
                "is_within_budget": within_budget_ratio,
                "is_over_budget": over_budget_ratio,
                "within_budget_ratio": within_budget_ratio,
                "over_budget_ratio": over_budget_ratio,
                "num_steps": meta.get("num_steps"),
                "is_correct": is_correct,
                "ground_truth": meta.get("ground_truth"),
                "extracted_answer": meta.get("extracted_answer"),
                "forced_conclude_count": meta.get("forced_conclude_count"),
                "status": str(sample.status),
                "response_length": sample.response_length,
                "reward": reward,
                "first_chosen_action": meta.get("first_chosen_action"),
                "first_chosen_starter": meta.get("first_chosen_starter"),
                "answer_text": meta.get("answer_text") or "",
                "reasoning_trace_chars": len(meta.get("reasoning_trace") or ""),
            })

        present_rewards = [r for r in rewards if r is not None]
        distinct_starters = len({s for s in chosen_starters if s})
        group_data["correctness"] = correctness
        group_data["rewards"] = rewards
        group_data["chosen_actions"] = chosen_actions
        group_data["chosen_starters"] = chosen_starters
        group_data["distinct_starters"] = distinct_starters
        group_data["reward_mean"] = float(np.mean(present_rewards)) if present_rewards else None
        group_data["reward_std"] = float(np.std(present_rewards)) if present_rewards else None
        group_data["zero_reward_variance"] = (
            bool(np.std(present_rewards) <= 1e-12) if len(present_rewards) >= 2 else None
        )
        group_data["correctness_sum"] = float(sum(correctness))
        group_data["has_variance"] = len(set(present_rewards)) > 1 if present_rewards else False
        group_data["all_correct"] = all(c >= 1.0 - 1e-9 for c in correctness) if correctness else False
        group_data["all_incorrect"] = all(c <= 1e-9 for c in correctness) if correctness else False
        group_data["avg_budget_used_pct"] = _mean_or_none(budget_used_pcts)
        group_data["avg_budget_remaining_pct"] = _mean_or_none(budget_remaining_pcts)
        group_data["avg_num_steps"] = _mean_or_none(num_steps)
        group_data["avg_response_length"] = _mean_or_none(response_lengths)
        dump["groups"].append(group_data)

    dump["summary"] = {
        "groups_with_variance": sum(1 for g in dump["groups"] if g["has_variance"]),
        "groups_zero_reward_variance": sum(
            1 for g in dump["groups"] if g["zero_reward_variance"] is True
        ),
        "groups_all_correct": sum(1 for g in dump["groups"] if g["all_correct"]),
        "groups_all_incorrect": sum(1 for g in dump["groups"] if g["all_incorrect"]),
        "avg_group_budget_used_pct": _mean_or_none([g["avg_budget_used_pct"] for g in dump["groups"]]),
        "avg_group_num_steps": _mean_or_none([g["avg_num_steps"] for g in dump["groups"]]),
        "avg_group_reward_std": _mean_or_none([g["reward_std"] for g in dump["groups"]]),
        "avg_group_distinct_starters": _mean_or_none(
            [float(g["distinct_starters"]) for g in dump["groups"]]
        ),
    }

    try:
        with open(dump_path, "w") as f:
            json.dump(dump, f, indent=2, default=str)
        logger.info(
            "[acts_log] Dumped first rollout batch (%d groups) to %s; "
            "all_correct=%d all_incorrect=%d with_variance=%d",
            len(groups),
            dump_path,
            dump["summary"]["groups_all_correct"],
            dump["summary"]["groups_all_incorrect"],
            dump["summary"]["groups_with_variance"],
        )
    except Exception as e:
        logger.warning(f"[acts_log] Failed to dump first batch: {e}")


def log_rollout_data(
    rollout_id: int,
    args: Any,
    samples: list,
    rollout_extra_metrics: dict | None,
    rollout_time: float,
) -> bool:
    """Custom log function for ACTS rollouts.

    Returns False so SLIME also runs its default logging.
    """
    if not samples:
        return False

    flat_samples: list = []
    for s in samples:
        if isinstance(s, list):
            flat_samples.extend(s)
        else:
            flat_samples.append(s)

    if not flat_samples:
        return False

    global _DUMP_DONE
    if not _DUMP_DONE:
        _DUMP_DONE = True
        _dump_first_batch(args, samples, flat_samples, rollout_id)

    # --- Per-sample (flat) metrics ---
    correctness: list[float] = []
    rewards: list[float] = []
    num_steps_list: list[float] = []
    budget_used_pcts: list[float] = []
    budget_remained_pcts: list[float] = []
    within_budget_flags: list[float] = []
    over_budget_flags: list[float] = []
    response_cap_flags: list[float] = []
    trace_cap_flags: list[float] = []
    budget_penalties: list[float] = []
    correct_over_budget_penalties: list[float] = []
    wrong_under_budget_penalties: list[float] = []
    wrong_over_budget_penalties: list[float] = []

    for sample in flat_samples:
        meta = sample.metadata or {}
        total_budget = meta.get("total_budget", 1)
        real_budget_remaining = meta.get("real_budget_remaining", 0.0)
        is_correct = _safe_float(meta.get("is_correct")) or 0.0
        num_steps = meta.get("num_steps", 0)
        within_budget_ratio, over_budget_ratio = _resolve_budget_ratios(meta)

        budget_frac = real_budget_remaining / float(total_budget) if total_budget > 0 else 0.0
        used_frac = 1.0 - budget_frac

        correctness.append(float(is_correct))
        reward_val = _safe_float(sample.reward)
        if reward_val is not None:
            rewards.append(reward_val)
        num_steps_list.append(float(num_steps))
        budget_used_pcts.append(max(0.0, 100.0 * used_frac))
        budget_remained_pcts.append(max(0.0, min(100.0, 100.0 * budget_frac)))
        if within_budget_ratio is not None:
            within_budget_flags.append(within_budget_ratio)
        if over_budget_ratio is not None:
            over_budget_flags.append(over_budget_ratio)
        response_cap_flags.append(float(bool(meta.get("response_cap_reached"))))
        trace_cap_flags.append(float(bool(meta.get("trace_cap_reached"))))
        for key, target in [
            ("budget_penalty_applied", budget_penalties),
            ("correct_over_budget_penalty", correct_over_budget_penalties),
            ("wrong_under_budget_penalty", wrong_under_budget_penalties),
            ("wrong_over_budget_penalty", wrong_over_budget_penalties),
        ]:
            value = _safe_float(meta.get(key))
            if value is not None:
                target.append(value)

    # --- Within-group metrics ---
    groups = _reconstruct_groups(args, samples, flat_samples)
    per_group_reward_std: list[float] = []
    per_group_zero_var: list[float] = []
    per_group_action_mean_reward_std: list[float] = []
    per_group_nonzero_action_advantage: list[float] = []
    per_group_distinct_actions: list[float] = []
    per_group_distinct_starters: list[float] = []
    per_group_correctness_variance: list[float] = []
    per_group_budget_only_variance: list[float] = []
    for group in groups:
        correctness_signal, budget_only_signal = _group_signal_flags(group)
        per_group_correctness_variance.append(correctness_signal)
        per_group_budget_only_variance.append(budget_only_signal)

        group_rewards = [_safe_float(s.reward) for s in group]
        group_rewards = [r for r in group_rewards if r is not None]
        if len(group_rewards) >= 2:
            group_reward_std = float(np.std(group_rewards))
            per_group_reward_std.append(group_reward_std)
            per_group_zero_var.append(1.0 if group_reward_std <= 1e-12 else 0.0)
        elif len(group_rewards) == 1:
            per_group_reward_std.append(0.0)

        action_mean_rewards = _action_mean_rewards_for_group(group)
        if len(action_mean_rewards) >= 2:
            action_mean_reward_std = float(np.std(action_mean_rewards))
            per_group_action_mean_reward_std.append(action_mean_reward_std)
            per_group_nonzero_action_advantage.append(
                1.0 if action_mean_reward_std > 1e-12 else 0.0
            )
        elif len(action_mean_rewards) == 1:
            per_group_action_mean_reward_std.append(0.0)
            per_group_nonzero_action_advantage.append(0.0)

        actions = {
            (s.metadata or {}).get("first_chosen_action", "")
            for s in group
            if (s.metadata or {}).get("first_chosen_action")
        }
        starters = {
            (s.metadata or {}).get("first_chosen_starter", "")
            for s in group
            if (s.metadata or {}).get("first_chosen_starter")
        }
        if len(group) > 0:
            per_group_distinct_actions.append(float(len(actions)))
            per_group_distinct_starters.append(float(len(starters)))

    # --- Build log dict ---
    step = compute_rollout_step(args, rollout_id)
    log_dict: dict[str, Any] = {
        "controller/rollout_step": rollout_id,
        "controller/correctness": float(np.mean(correctness)) if correctness else 0.0,
        "controller/num_steps": float(np.mean(num_steps_list)) if num_steps_list else 0.0,
        "controller/budget_used_pct": float(np.mean(budget_used_pcts)) if budget_used_pcts else 0.0,
        "controller/budget_remained_pct": float(np.mean(budget_remained_pcts)) if budget_remained_pcts else 0.0,
        "rollout/step": step,
    }
    if rewards:
        log_dict["controller/rewards"] = float(np.mean(rewards))
    if per_group_reward_std:
        log_dict["controller/reward_std_within_group"] = float(np.mean(per_group_reward_std))
    if per_group_zero_var:
        log_dict["controller/zero_var_group_ratio"] = float(np.mean(per_group_zero_var))
    if per_group_action_mean_reward_std:
        log_dict["controller/action_mean_reward_std_within_group"] = float(
            np.mean(per_group_action_mean_reward_std)
        )
    if per_group_nonzero_action_advantage:
        log_dict["controller/nonzero_action_advantage_group_ratio"] = float(
            np.mean(per_group_nonzero_action_advantage)
        )
    if per_group_correctness_variance:
        log_dict["controller/correctness_variance_group_ratio"] = float(
            np.mean(per_group_correctness_variance)
        )
    if per_group_budget_only_variance:
        log_dict["controller/budget_only_variance_group_ratio"] = float(
            np.mean(per_group_budget_only_variance)
        )
    if per_group_distinct_actions:
        log_dict["controller/action_diversity_within_group"] = float(
            np.mean(per_group_distinct_actions)
        )
    if per_group_distinct_starters:
        log_dict["controller/starter_diversity_within_group"] = float(
            np.mean(per_group_distinct_starters)
        )
    if within_budget_flags:
        log_dict["controller/within_budget_ratio"] = float(np.mean(within_budget_flags))
    if over_budget_flags:
        log_dict["controller/over_budget_ratio"] = float(np.mean(over_budget_flags))
    if response_cap_flags:
        log_dict["controller/response_cap_ratio"] = float(np.mean(response_cap_flags))
    if trace_cap_flags:
        log_dict["controller/trace_cap_ratio"] = float(np.mean(trace_cap_flags))
    if budget_penalties:
        log_dict["controller/budget_penalty_applied"] = float(np.mean(budget_penalties))
    if correct_over_budget_penalties:
        log_dict["controller/correct_over_budget_penalty"] = float(
            np.mean(correct_over_budget_penalties)
        )
    if wrong_under_budget_penalties:
        log_dict["controller/wrong_under_budget_penalty"] = float(
            np.mean(wrong_under_budget_penalties)
        )
    if wrong_over_budget_penalties:
        log_dict["controller/wrong_over_budget_penalty"] = float(
            np.mean(wrong_over_budget_penalties)
        )

    logger.info(f"controller metrics {rollout_id}: {log_dict}")
    logging_utils.log(args, log_dict, step_key="controller/rollout_step")
    _log_sample_group_table(args, groups, rollout_id, step)

    return False
