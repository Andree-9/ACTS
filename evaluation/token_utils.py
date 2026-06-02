"""Utilities for splitting model output into thinking vs answer tokens."""

from __future__ import annotations

from typing import Tuple


def split_thinking_and_answer(
    model_output: str,
    prompt_has_closed_think: bool = False,
) -> Tuple[str, str]:
    """Split output into thinking and answer portions.

    Handles both cases:
      - <think> and </think> both present in output
      - Only </think> present (when <think> was injected by the chat template)
      - No tags at all:
        - answer-only continuation when the prompt already closed </think>
        - otherwise, entire output treated as thinking
    """
    think_start = model_output.find("<think>")
    think_end = model_output.find("</think>")

    if think_start != -1 and think_end != -1 and think_end > think_start:
        thinking = model_output[think_start : think_end + len("</think>")]
        answer = model_output[think_end + len("</think>") :]
    elif think_start != -1 and think_end == -1:
        # Thinking started but never closed (hit max tokens)
        thinking = model_output[think_start:]
        answer = ""
    elif think_start == -1 and think_end != -1:
        # <think> was in prompt, output starts with thinking content
        thinking = model_output[: think_end + len("</think>")]
        answer = model_output[think_end + len("</think>") :]
    elif prompt_has_closed_think:
        # NoThinking prefills a closed think block in the prompt, so the model
        # continues directly in the final-answer channel.
        thinking = ""
        answer = model_output
    else:
        # No think tags — treat entire output as thinking
        thinking = model_output
        answer = ""

    return thinking, answer


def count_think_answer_tokens(
    tokenizer,
    model_output: str,
    prompt_has_closed_think: bool = False,
) -> Tuple[int, int]:
    """Count tokens separately for thinking and answer portions."""
    thinking, answer = split_thinking_and_answer(
        model_output,
        prompt_has_closed_think=prompt_has_closed_think,
    )
    t = len(tokenizer.encode(thinking, add_special_tokens=False)) if thinking else 0
    a = len(tokenizer.encode(answer, add_special_tokens=False)) if answer else 0
    return t, a
