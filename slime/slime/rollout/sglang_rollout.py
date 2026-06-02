import asyncio
import copy
import importlib
import inspect
import logging
import uuid
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import numpy as np
import pybase64
import sglang_router
from packaging.version import parse
from tqdm import tqdm

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.http_utils import get, post
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.processing_utils import (
    build_processor_kwargs,
    encode_image_for_rollout_engine,
    load_processor,
    load_tokenizer,
)
from slime.utils.types import Sample

from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout", "get_model_url"]

logger = logging.getLogger(__name__)


def _first_sample_in_group(group: list[Sample] | list[list[Sample]]) -> Sample:
    return group[0][0] if isinstance(group[0], list) else group[0]


def _group_sort_key(group: list[Sample] | list[list[Sample]]) -> int:
    return _first_sample_in_group(group).index


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _controller_budget_stats(meta: dict[str, Any]) -> tuple[float | None, float | None]:
    total_budget = _safe_float(meta.get("total_budget"))
    remaining_budget = _safe_float(meta.get("real_budget_remaining"))
    if total_budget is None or remaining_budget is None or total_budget <= 0:
        return None, None
    remaining_pct = max(0.0, min(100.0, 100.0 * remaining_budget / total_budget))
    used_pct = max(0.0, 100.0 - remaining_pct)
    return used_pct, remaining_pct


def _controller_budget_ratios(meta: dict[str, Any]) -> tuple[float | None, float | None]:
    mc_is_over_budget = meta.get("mc_is_over_budget")
    if isinstance(mc_is_over_budget, list) and mc_is_over_budget:
        over_budget_ratio = float(np.mean([1.0 if bool(value) else 0.0 for value in mc_is_over_budget]))
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
    over_budget = (
        bool(meta.get("trace_cap_reached"))
        or remaining_budget / total_budget < -over_budget_grace_frac
    )
    return float(not over_budget), float(over_budget)


def _controller_action_key(meta: dict[str, Any]) -> str | None:
    action = str(meta.get("first_chosen_action") or "").strip().lower()
    return action or None


def _controller_action_mean_rewards_for_group(group: list[Sample]) -> list[float]:
    raw_rewards: list[float] = []
    action_to_indices: dict[str, list[int]] = {}

    for sample in group:
        reward = _safe_float(sample.reward)
        if reward is None:
            continue
        raw_rewards.append(reward)

        action = _controller_action_key(sample.metadata or {})
        if action is not None:
            action_to_indices.setdefault(action, []).append(len(raw_rewards) - 1)

    action_mean_rewards = list(raw_rewards)
    for indices in action_to_indices.values():
        mean_reward = float(np.mean([raw_rewards[idx] for idx in indices]))
        for idx in indices:
            action_mean_rewards[idx] = mean_reward

    return action_mean_rewards


def _controller_group_signal_flags(group: list[Sample]) -> tuple[float, float]:
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


def _mean_metric(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _controller_metrics_from_groups(groups: list[list[Sample]], prefix: str) -> dict[str, float]:
    flat_samples = [sample for group in groups for sample in group]
    metrics: dict[str, float] = {}
    if not flat_samples:
        return metrics
    metrics[f"{prefix}/group_count"] = float(len(groups))
    metrics[f"{prefix}/sample_count"] = float(len(flat_samples))

    correctness: list[float] = []
    rewards: list[float] = []
    num_steps: list[float] = []
    budget_used_pcts: list[float] = []
    budget_remained_pcts: list[float] = []
    within_budget_flags: list[float] = []
    over_budget_flags: list[float] = []
    response_cap_flags: list[float] = []
    trace_cap_flags: list[float] = []
    budget_penalties: list[float] = []
    correct_over_budget_penalties: list[float] = []
    wrong_under_budget_penalties: list[float] = []

    for sample in flat_samples:
        meta = sample.metadata or {}
        correctness.append(float(_safe_float(meta.get("is_correct")) or 0.0))

        reward = _safe_float(sample.reward)
        if reward is not None:
            rewards.append(reward)

        step_count = _safe_float(meta.get("num_steps"))
        if step_count is not None:
            num_steps.append(step_count)

        used_pct, remained_pct = _controller_budget_stats(meta)
        if used_pct is not None:
            budget_used_pcts.append(used_pct)
        if remained_pct is not None:
            budget_remained_pcts.append(remained_pct)

        within_budget_ratio, over_budget_ratio = _controller_budget_ratios(meta)
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
        ]:
            value = _safe_float(meta.get(key))
            if value is not None:
                target.append(value)

    per_group_reward_std: list[float] = []
    per_group_zero_var: list[float] = []
    per_group_action_mean_reward_std: list[float] = []
    per_group_nonzero_action_advantage: list[float] = []
    per_group_distinct_actions: list[float] = []
    per_group_distinct_starters: list[float] = []
    per_group_correctness_variance: list[float] = []
    per_group_budget_only_variance: list[float] = []

    for group in groups:
        correctness_signal, budget_only_signal = _controller_group_signal_flags(group)
        per_group_correctness_variance.append(correctness_signal)
        per_group_budget_only_variance.append(budget_only_signal)

        group_rewards = [_safe_float(sample.reward) for sample in group]
        group_rewards = [reward for reward in group_rewards if reward is not None]
        if len(group_rewards) >= 2:
            reward_std = float(np.std(group_rewards))
            per_group_reward_std.append(reward_std)
            per_group_zero_var.append(1.0 if reward_std <= 1e-12 else 0.0)
        elif len(group_rewards) == 1:
            per_group_reward_std.append(0.0)

        action_mean_rewards = _controller_action_mean_rewards_for_group(group)
        if len(action_mean_rewards) >= 2:
            action_reward_std = float(np.std(action_mean_rewards))
            per_group_action_mean_reward_std.append(action_reward_std)
            per_group_nonzero_action_advantage.append(1.0 if action_reward_std > 1e-12 else 0.0)
        elif len(action_mean_rewards) == 1:
            per_group_action_mean_reward_std.append(0.0)
            per_group_nonzero_action_advantage.append(0.0)

        actions = {
            (sample.metadata or {}).get("first_chosen_action", "")
            for sample in group
            if (sample.metadata or {}).get("first_chosen_action")
        }
        starters = {
            (sample.metadata or {}).get("first_chosen_starter", "")
            for sample in group
            if (sample.metadata or {}).get("first_chosen_starter")
        }
        if group:
            per_group_distinct_actions.append(float(len(actions)))
            per_group_distinct_starters.append(float(len(starters)))

    values = {
        "correctness": _mean_metric(correctness),
        "num_steps": _mean_metric(num_steps),
        "budget_used_pct": _mean_metric(budget_used_pcts),
        "budget_remained_pct": _mean_metric(budget_remained_pcts),
        "rewards": _mean_metric(rewards),
        "reward_std_within_group": _mean_metric(per_group_reward_std),
        "zero_var_group_ratio": _mean_metric(per_group_zero_var),
        "action_mean_reward_std_within_group": _mean_metric(per_group_action_mean_reward_std),
        "nonzero_action_advantage_group_ratio": _mean_metric(per_group_nonzero_action_advantage),
        "correctness_variance_group_ratio": _mean_metric(per_group_correctness_variance),
        "budget_only_variance_group_ratio": _mean_metric(per_group_budget_only_variance),
        "action_diversity_within_group": _mean_metric(per_group_distinct_actions),
        "starter_diversity_within_group": _mean_metric(per_group_distinct_starters),
        "within_budget_ratio": _mean_metric(within_budget_flags),
        "over_budget_ratio": _mean_metric(over_budget_flags),
        "response_cap_ratio": _mean_metric(response_cap_flags),
        "trace_cap_ratio": _mean_metric(trace_cap_flags),
        "budget_penalty_applied": _mean_metric(budget_penalties),
        "correct_over_budget_penalty": _mean_metric(correct_over_budget_penalties),
        "wrong_under_budget_penalty": _mean_metric(wrong_under_budget_penalties),
    }
    for name, value in values.items():
        if value is not None:
            metrics[f"{prefix}/{name}"] = value
    return metrics


def get_model_url(args: Namespace, model_name: str, endpoint: str = "/generate") -> str:
    """Return the router URL for a named model.

    Use this in custom rollout functions to route requests to a specific
    model when multiple models are deployed via ``--sglang-config``::

        url = get_model_url(args, "ref", "/generate")
        resp = await post(url, json=payload)

    Falls back to the default router if *model_name* is not found or
    ``sglang_model_routers`` is not set.
    """
    routers = getattr(args, "sglang_model_routers", None)
    if routers and model_name in routers:
        ip, port = routers[model_name]
        return f"http://{ip}:{port}{endpoint}"
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}{endpoint}"


class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        self.reset()

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate using traditional SGLang router with token-based workflow"""
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    if state.processor and sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
        processor_kwargs = build_processor_kwargs(sample.multimodal_inputs)
        processor_output = state.processor(text=sample.prompt, **processor_kwargs)
        prompt_ids = processor_output["input_ids"][0]
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)

    if len(sample.response) > 0:
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    has_multimodal = sample.multimodal_inputs and sample.multimodal_inputs.get("images")
    if has_multimodal:
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    # Use existing tokens for multi-turn or tokenize the new prompt
    if len(sample.response) > 0:
        payload["input_ids"] = sample.tokens
    elif has_multimodal:
        # For multimodal first-turn: send text so SGLang handles image token
        # expansion internally (the processor-expanded input_ids have N patch
        # tokens per image which would mismatch the image_data count).
        payload["text"] = sample.prompt
        if not sample.tokens:
            sample.tokens = prompt_ids
    else:
        payload["input_ids"] = prompt_ids
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_ids

    # Use session_id for consistent hashing routing (SGLang Model Gateway)
    headers = None
    if sample.session_id:
        if getattr(args, "router_policy", None) == "consistent_hashing":
            headers = {"X-SMG-Routing-Key": sample.session_id}

    output = await post(url, payload, headers=headers)

    if "output_token_logprobs" in output["meta_info"]:
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_response_tokens, new_response_log_probs = [], []

    # Update sample with tokens directly - avoiding re-tokenization
    sample.tokens = sample.tokens + new_response_tokens
    sample.response_length += len(new_response_tokens)
    sample.response += output["text"]

    # When partial rollout and masking off policy is enabled, update the loss mask
    if sample.loss_mask is not None:
        assert args.partial_rollout and args.mask_offpolicy_in_partial_rollout
        sample.loss_mask += [1] * len(new_response_tokens)

    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs += new_response_log_probs

    if "routed_experts" in output["meta_info"]:
        sample.rollout_routed_experts = np.frombuffer(
            pybase64.b64decode(output["meta_info"]["routed_experts"].encode("ascii")),
            dtype=np.int32,
        ).reshape(
            len(sample.tokens) - 1,
            args.num_layers,
            args.moe_router_topk,
        )

    sample.update_from_meta_info(args, output["meta_info"])

    return sample


async def generate_and_rm(
    args: Namespace,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        # Check sample.generate_function_path for per-sample custom_generate_function_path (e.g., from eval dataset config)
        custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

        if custom_func_path is not None:
            custom_generate_func = load_function(custom_func_path)
            # if signature has evaluation, pass evaluation
            if "evaluation" in inspect.signature(custom_generate_func).parameters:
                sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
            else:
                sample = await custom_generate_func(args, sample, sampling_params)
        else:
            sample = await generate(args, sample, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        if sample.reward is None:
            sample.reward = await async_rm(args, sample)

    return sample


async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    state = GenerateState(args)

    if state.aborted:
        return group

    # Generate a unique session_id for each sample in the group
    for sample in group:
        if sample.session_id is None:
            sample.session_id = str(uuid.uuid4())

    custom_func_path = getattr(group[0], "generate_function_path", None) or args.custom_generate_function_path
    if custom_func_path is not None and not evaluation:
        custom_generate_func = load_function(custom_func_path)
        custom_generate_group_func = getattr(
            importlib.import_module(custom_generate_func.__module__),
            "generate_group",
            None,
        )
        if custom_generate_group_func is not None:
            if "evaluation" in inspect.signature(custom_generate_group_func).parameters:
                group = await custom_generate_group_func(args, group, sampling_params, evaluation=evaluation)
            else:
                group = await custom_generate_group_func(args, group, sampling_params)

            if not state.aborted and args.group_rm:
                rewards = await batched_async_rm(args, group)
                for sample, reward in zip(group, rewards, strict=False):
                    sample.reward = reward
            elif not state.aborted:
                samples_need_reward = [sample for sample in group if sample.reward is None]
                if samples_need_reward:
                    rewards = await batched_async_rm(args, samples_need_reward)
                    for sample, reward in zip(samples_need_reward, rewards, strict=False):
                        sample.reward = reward

            return group

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(
            asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))
        )

    group = await asyncio.gather(*tasks)

    # for the rm that need the whole group, we will do the rm here
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    return group


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True

    if parse(sglang_router.__version__) <= parse("0.2.1"):
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

    logger.info(f"Abort request for {urls}")
    abort_tasks = [post(f"{url}/abort_request", {"abort_all": True}) for url in urls]
    abort_results = await asyncio.gather(*abort_tasks, return_exceptions=True)
    for url, result in zip(urls, abort_results, strict=False):
        if isinstance(result, Exception):
            logger.warning(f"Failed to abort worker at {url}: {result}")

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    return aborted_samples


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    data = []
    all_data = []
    do_print = True
    if getattr(args, "soft_dynamic_sampling_filter", False):
        candidate_limit = max(target_data_size, args.over_sampling_batch_size)
        max_inflight_groups = min(target_data_size, candidate_limit)
        submitted_groups = 0
        completed_groups = 0
        passed_groups = []
        filtered_groups = []

        def submit_candidate_groups(num_groups: int) -> int:
            nonlocal submitted_groups
            num_groups = min(num_groups, candidate_limit - submitted_groups)
            if num_groups <= 0:
                return 0
            samples = data_source(num_groups)
            state.submit_generate_tasks(samples)
            submitted_groups += len(samples)
            return len(samples)

        submit_candidate_groups(max_inflight_groups)

        pbar = tqdm(
            total=candidate_limit * args.n_samples_per_prompt,
            desc="Soft rollout generation",
        )
        while state.pendings and completed_groups < candidate_limit:
            done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                group: list[Sample] = task.result()

                if do_print:
                    sample = _first_sample_in_group(group)
                    logger.info(
                        f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
                    )
                    do_print = False

                assert len(group) == args.n_samples_per_prompt
                all_data.append(group)
                dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
                if dynamic_filter_output.keep:
                    passed_groups.append(group)
                else:
                    filtered_groups.append((dynamic_filter_output.reason, group))
                completed_groups += 1
                pbar.update(args.n_samples_per_prompt)

            if len(passed_groups) >= target_data_size:
                break
            if completed_groups >= candidate_limit:
                break
            submit_candidate_groups(len(done))

        pbar.close()

        aborted_samples = await abort(args, rollout_id) if state.pendings else []

        passed_groups = sorted(passed_groups, key=_group_sort_key)
        filtered_groups = sorted(filtered_groups, key=lambda item: _group_sort_key(item[1]))

        data = passed_groups[:target_data_size]
        zero_fill_needed = max(0, target_data_size - len(data))
        zero_fill_groups = [group for _, group in filtered_groups[:zero_fill_needed]]
        data.extend(zero_fill_groups)

        for reason, _ in filtered_groups[zero_fill_needed:]:
            metric_gatherer.on_dynamic_filter_drop(reason=reason)

        assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
        data = sorted(data, key=_group_sort_key)
        all_samples = sorted(all_data, key=_group_sort_key)

        sample = _first_sample_in_group(data[-1])
        logger.info(
            f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
        )

        state.reset()
        if args.rollout_sample_filter_path is not None:
            filter_func = load_function(args.rollout_sample_filter_path)
            filter_func(args, data)

        if args.rollout_all_samples_process_path is not None:
            process_func = load_function(args.rollout_all_samples_process_path)
            process_func(args, all_samples, data_source)

        metrics = metric_gatherer.collect()
        metrics.update(_controller_metrics_from_groups(all_data, "controller_raw"))

        return RolloutFnTrainOutput(samples=data, metrics=metrics), aborted_samples

    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                sample = _first_sample_in_group(group)
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = _first_sample_in_group(data[-1])
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=_group_sort_key)
    all_samples = sorted(all_data, key=_group_sort_key)

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    coros = []
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    results_list = await asyncio.gather(*coros)
    results = {}
    for r in results_list:
        results.update(r)
    return RolloutFnEvalOutput(data=results), []


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            logger.info(
                "eval_rollout_single_dataset example data: "
                f"{[str(sample.prompt) + sample.response]} "
                f"reward={sample.reward}"
            )
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to get and store samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        RolloutFnTrainOutput | RolloutFnEvalOutput: the output of the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    data_source.add_samples(aborted_samples)
    return output
