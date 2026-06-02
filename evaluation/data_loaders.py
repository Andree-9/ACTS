"""
Dataset loading and prompt formatting for reasoning benchmarks.

Reuses dataset configs from old/evaluation but always applies the model's own
chat template via tokenizer.apply_chat_template() — no hardcoded fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from datasets import load_dataset


@dataclass
class EvalSample:
    """A single evaluation sample with prompt and ground truth."""
    doc_id: int
    problem: str
    answer: str
    formatted_prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset configs — same benchmarks as old/evaluation/data_loaders.py
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    "math500": {
        "path": "simplescaling/openaimath",
        "split": "test",
        "problem_key": "problem",
        "answer_key": "answer",
        "max_gen_toks": 32768,
    },
    "aime2024": {
        "path": "simplescaling/aime24_nofigures",
        "split": "train",
        "problem_key": "problem",
        "answer_key": "answer",
        "max_gen_toks": 32768,
    },
    "amc": {
        "path": "AI-MO/aimo-validation-amc",
        "split": "train",
        "problem_key": "problem",
        "answer_key": "answer",
        "max_gen_toks": 32768,
    },
    "olympiadbench": {
        "path": "FUfu99/OlympiadBench_maths_origin",
        "split": "test",
        "problem_key": "problem",
        "answer_key": "final_answer",
        "max_gen_toks": 32768,
    },
    # Non-math: graduate-level science MCQ. `problem` already contains the
    # question, the (A)/(B)/(C)/(D) options, and an instruction to emit the
    # final answer as \boxed{LETTER}. `solution` is stored as \boxed{LETTER};
    # we extract the bare letter at load time to match the plain-string
    # convention used by the math benchmarks.
    "gpqa_diamond": {
        "path": "hendrydong/gpqa_diamond_mc",
        "split": "test",
        "problem_key": "problem",
        "answer_key": "solution",
        "answer_is_boxed": True,
        "max_gen_toks": 32768,
    },
}


def format_prompt(problem: str, tokenizer) -> str:
    """Format a problem using the model's own chat template.

    Always delegates to tokenizer.apply_chat_template() so the prompt matches
    whatever the model was trained with (DeepSeek native, ChatML, Qwen3, etc.).
    """
    messages = [{"role": "user", "content": problem}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def load_benchmark(
    benchmark_name: str,
    tokenizer,
    limit: Optional[int] = None,
) -> List[EvalSample]:
    """Load a benchmark dataset with chat-template-formatted prompts."""
    if benchmark_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )

    cfg = DATASET_CONFIGS[benchmark_name]
    dataset = load_dataset(cfg["path"], split=cfg["split"])
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    answer_is_boxed = bool(cfg.get("answer_is_boxed", False))
    if answer_is_boxed:
        from .scoring import extract_boxed_answer

    samples: list[EvalSample] = []
    for idx, doc in enumerate(dataset):
        problem = doc[cfg["problem_key"]]
        answer = doc[cfg["answer_key"]]
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        answer = str(answer)
        if answer_is_boxed:
            answer = extract_boxed_answer(answer) or answer

        formatted = (
            format_prompt(problem, tokenizer)
            if tokenizer is not None
            else ""
        )
        samples.append(EvalSample(
            doc_id=idx,
            problem=problem,
            answer=answer,
            formatted_prompt=formatted,
            metadata={"benchmark": benchmark_name, "max_gen_toks": cfg["max_gen_toks"]},
        ))

    return samples
