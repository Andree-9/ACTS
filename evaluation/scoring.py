"""
Answer extraction and scoring for reasoning benchmarks.

The math_verify path intentionally grades only the final boxed answer.
This matches the common evaluation convention for math reasoning tasks and
avoids rewarding intermediate expressions from the reasoning trace.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import queue
import threading
from typing import Dict, List, Optional, Sequence, Tuple

from math_verify.grader import verify as math_verify
from math_verify.parser import LatexExtractionConfig, parse

logger = logging.getLogger(__name__)

API_FREE_BACKENDS = {"exact", "math_verify"}
_BOXED_EXTRACTION_CONFIG = [LatexExtractionConfig()]


def _read_timeout_env(name: str, default: int) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; falling back to %s", name, raw, default)
        return default
    return None if value <= 0 else value


def _read_worker_timeout_env(name: str, default: float) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; falling back to %s", name, raw, default)
        return default
    return None if value <= 0 else value


_MATH_VERIFY_PARSE_TIMEOUT_SECONDS = _read_timeout_env(
    "AGENTICCOT_MATH_VERIFY_PARSE_TIMEOUT_SECONDS",
    5,
)
_MATH_VERIFY_VERIFY_TIMEOUT_SECONDS = _read_timeout_env(
    "AGENTICCOT_MATH_VERIFY_VERIFY_TIMEOUT_SECONDS",
    5,
)
_MATH_VERIFY_WORKER_TIMEOUT_SECONDS = _read_worker_timeout_env(
    "AGENTICCOT_MATH_VERIFY_WORKER_TIMEOUT_SECONDS",
    15.0,
)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------
def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]
    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def _response_region_for_scoring(output: str) -> str:
    """
    Prefer the post-`</think>` answer segment when it exists.

    Direct eval may return a single string containing both the thinking trace
    and the final answer. Controlled eval stores the final answer separately,
    but some callers still concatenate the two segments for logging. In both
    cases, boxed-answer scoring should inspect the final answer channel first.
    """
    text = str(output or "")
    if "</think>" in text:
        answer_region = text.rsplit("</think>", 1)[-1].strip()
        if answer_region:
            return answer_region
    return text


def extract_boxed_answer(output: str) -> str:
    """Return the final boxed answer content, or an empty string."""
    if not output:
        return ""
    boxed = last_boxed_only_string(_response_region_for_scoring(output))
    if boxed is None:
        return ""
    try:
        return str(remove_boxed(boxed)).strip()
    except Exception:
        return ""


def extract_answer_candidate(output: str) -> str:
    """Return the answer candidate that is actually graded."""
    return extract_boxed_answer(output)


def normalize_digit_pair(prediction: str, ground_truth: str) -> Tuple[str, str]:
    prediction = str(prediction).strip()
    ground_truth = str(ground_truth).strip()
    if prediction.isdigit() and ground_truth.isdigit():
        prediction = str(int(prediction))
        ground_truth = str(int(ground_truth))
    return prediction, ground_truth


def _boxed_text_for_scoring(text: str) -> Optional[str]:
    boxed = last_boxed_only_string(str(text))
    if boxed is not None:
        return boxed

    stripped = str(text).strip()
    if not stripped:
        return None
    return f"\\boxed{{{stripped}}}"


def _math_verify_boxed_match_inline(
    prediction_boxed: str,
    ground_truth_boxed: str,
    parse_timeout_seconds: int | None,
    verify_timeout_seconds: int | None,
) -> bool:
    """Compare two boxed answers with math_verify on the current thread."""
    try:
        gold_parsed = parse(
            ground_truth_boxed,
            extraction_config=_BOXED_EXTRACTION_CONFIG,
            parsing_timeout=parse_timeout_seconds,
        )
        pred_parsed = parse(
            prediction_boxed,
            extraction_config=_BOXED_EXTRACTION_CONFIG,
            parsing_timeout=parse_timeout_seconds,
        )
    except Exception:
        logger.debug("math_verify parse failed", exc_info=True)
        return False

    if not gold_parsed or not pred_parsed:
        return False

    try:
        return bool(
            math_verify(
                gold_parsed,
                pred_parsed,
                timeout_seconds=verify_timeout_seconds,
            )
        )
    except Exception:
        logger.debug("math_verify verify failed", exc_info=True)
        return False


def _math_verify_boxed_batch_worker(
    pairs: Sequence[Tuple[str, str]],
    parse_timeout_seconds: int | None,
    verify_timeout_seconds: int | None,
) -> List[bool]:
    return [
        _math_verify_boxed_match_inline(
            prediction_boxed,
            ground_truth_boxed,
            parse_timeout_seconds=parse_timeout_seconds,
            verify_timeout_seconds=verify_timeout_seconds,
        )
        for prediction_boxed, ground_truth_boxed in pairs
    ]


def _math_verify_boxed_batch_worker_entry(
    pairs: Sequence[Tuple[str, str]],
    parse_timeout_seconds: int | None,
    verify_timeout_seconds: int | None,
    result_queue,
) -> None:
    try:
        result_queue.put(
            _math_verify_boxed_batch_worker(
                pairs,
                parse_timeout_seconds=parse_timeout_seconds,
                verify_timeout_seconds=verify_timeout_seconds,
            )
        )
    except Exception:
        logger.exception("math_verify subprocess worker failed")
        result_queue.put(None)


def _math_verify_boxed_batch_subprocess(
    pairs: Sequence[Tuple[str, str]],
) -> List[bool]:
    timeout_result = [False] * len(pairs)
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_math_verify_boxed_batch_worker_entry,
        args=(
            list(pairs),
            _MATH_VERIFY_PARSE_TIMEOUT_SECONDS,
            _MATH_VERIFY_VERIFY_TIMEOUT_SECONDS,
            result_queue,
        ),
        daemon=True,
    )
    process.start()
    process.join(_MATH_VERIFY_WORKER_TIMEOUT_SECONDS)

    if process.is_alive():
        logger.warning(
            "math_verify subprocess timed out after %ss; marking %d sample(s) incorrect",
            _MATH_VERIFY_WORKER_TIMEOUT_SECONDS,
            len(pairs),
        )
        process.terminate()
        process.join()
        return timeout_result

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        logger.warning(
            "math_verify subprocess exited without a result (exitcode=%s); marking %d sample(s) incorrect",
            process.exitcode,
            len(pairs),
        )
        return timeout_result

    if result is None:
        return timeout_result
    return list(result)


def _score_boxed_pairs_math_verify(pairs: Sequence[Tuple[str, str]]) -> List[bool]:
    """
    Score boxed answers with math_verify.

    Direct and controlled eval run on the main thread, so math_verify's native
    signal-based timeouts are safe there. RL scoring runs on SLIME's background
    event-loop thread, so the same work is offloaded to a subprocess that can be
    terminated if a comparison stalls.
    """
    if not pairs:
        return []

    if threading.current_thread() is threading.main_thread():
        return _math_verify_boxed_batch_worker(
            pairs,
            parse_timeout_seconds=_MATH_VERIFY_PARSE_TIMEOUT_SECONDS,
            verify_timeout_seconds=_MATH_VERIFY_VERIFY_TIMEOUT_SECONDS,
        )
    return _math_verify_boxed_batch_subprocess(pairs)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_answer(
    extracted: str,
    ground_truth: str,
    backend: str = "math_verify",
    timeout_score: float = 0.0,
) -> bool:
    """Compare a pre-extracted answer against the ground truth."""
    del timeout_score

    backend = (backend or "math_verify").strip().lower()
    extracted, ground_truth = normalize_digit_pair(extracted, ground_truth)

    if backend == "exact":
        return extracted == ground_truth
    if backend != "math_verify":
        raise ValueError(f"Unsupported score backend: {backend}")
    if not extracted:
        return False

    pred_boxed = _boxed_text_for_scoring(extracted)
    gt_boxed = _boxed_text_for_scoring(ground_truth)
    if pred_boxed is None or gt_boxed is None:
        return False
    return _score_boxed_pairs_math_verify([(pred_boxed, gt_boxed)])[0]


def score_responses(
    model_outputs: Sequence[str],
    ground_truths: Sequence[str],
    backend: str = "math_verify",
    timeout_score: float = 0.0,
) -> List[bool]:
    """Score raw model responses in batch."""
    del timeout_score

    backend = (backend or "math_verify").strip().lower()
    if len(model_outputs) != len(ground_truths):
        raise ValueError("model_outputs and ground_truths must have the same length")

    if backend == "exact":
        return [
            score_answer(
                extract_answer_candidate(model_output),
                ground_truth,
                backend="exact",
            )
            for model_output, ground_truth in zip(model_outputs, ground_truths, strict=False)
        ]
    if backend != "math_verify":
        raise ValueError(f"Unsupported score backend: {backend}")

    indexed_pairs: List[Tuple[int, str, str]] = []
    results = [False] * len(model_outputs)
    for idx, (model_output, ground_truth) in enumerate(zip(model_outputs, ground_truths, strict=False)):
        extracted = extract_answer_candidate(model_output)
        if not extracted:
            continue
        pred_boxed = _boxed_text_for_scoring(extracted)
        gt_boxed = _boxed_text_for_scoring(ground_truth)
        if pred_boxed is None or gt_boxed is None:
            continue
        indexed_pairs.append((idx, pred_boxed, gt_boxed))

    if not indexed_pairs:
        return results

    batch_scores = _score_boxed_pairs_math_verify(
        [(pred_boxed, gt_boxed) for _, pred_boxed, gt_boxed in indexed_pairs]
    )
    for (idx, _, _), score in zip(indexed_pairs, batch_scores, strict=False):
        results[idx] = score
    return results


def score_response(
    model_output: str,
    ground_truth: str,
    backend: str = "math_verify",
    timeout_score: float = 0.0,
) -> bool:
    """Score a raw model response, grading only the final boxed answer."""
    return score_responses(
        [model_output],
        [ground_truth],
        backend=backend,
        timeout_score=timeout_score,
    )[0]


def process_results_api_free(
    doc: Dict,
    results: List[str],
    backend: str = "math_verify",
    timeout_score: float = 0.0,
) -> Dict:
    """API-free scoring (exact or math_verify). Compatible with old interface."""
    metrics: Dict = {"exact_match": None, "extracted_answers": []}

    gt = str(doc["answer"])
    if gt.isdigit():
        gt = str(int(gt))

    for i, raw_output in enumerate(results, start=1):
        extracted = extract_answer_candidate(raw_output)
        is_match = int(
            score_response(
                raw_output,
                gt,
                backend=backend,
                timeout_score=timeout_score,
            )
        )
        metrics["extracted_answers"].append(extracted)
        if i == 1:
            metrics["exact_match"] = is_match

    return metrics
