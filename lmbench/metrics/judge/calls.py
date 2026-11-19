"""Call LLM judges: one judge module once or n times, a judge function over many rows, and judge errors.

A judge call that still fails after all retries costs its row, not the run: ``CatchJudgeFailure`` (inline blocks) and
``judge_rows`` (per-row functions, optionally in parallel) log it and report the row as failed, so the metric can give
it the per-item status ``judge_failed``. Errors that would hit every row (``JUDGE_FATAL_ERRORS``) and all other
exceptions still end the run.
"""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import Any, Self

import dspy
from dspy.utils.exceptions import AdapterParseError, LMAuthError, LMBillingError, LMProviderError, LMTransportError
from litellm.exceptions import APIError as LiteLLMAPIError
from tqdm import tqdm

logger = logging.getLogger(__name__)

JUDGE_PARSE_RETRIES = 2

JUDGE_API_ERRORS: tuple[type[Exception], ...] = (LMTransportError, LMProviderError, LiteLLMAPIError)
"""A judge API call that failed after the LM's own retries (rate limit, timeout, 5xx, rejected request).

dspy 3 raises its own ``LMError`` subclasses for these, not litellm's ``APIError``.
"""
JUDGE_ERRORS: tuple[type[Exception], ...] = (AdapterParseError, *JUDGE_API_ERRORS)
"""A judge call that failed for one row: API error after retries or unparseable answer."""
JUDGE_FATAL_ERRORS: tuple[type[Exception], ...] = (LMAuthError, LMBillingError)
"""Judge errors that would hit every row (wrong key, quota): these end the run instead of failing one row."""


def _peek_last_finish_reason() -> str | None:
    """Best-effort finish_reason of the most recent dspy LM call (racy across threads)."""
    try:
        history = dspy.settings.lm.history
        if not history:
            return None
        response = history[-1].get("response")
        choices = getattr(response, "choices", None) or []
        if not choices:
            return None
        return getattr(choices[0], "finish_reason", None)
    except Exception:
        return None


def call_judge_n[JudgeResult](
    evaluator: Callable[..., JudgeResult],
    *,
    n: int,
    temperature: float,
    **kwargs: Any,
) -> list[JudgeResult]:
    """Call a DSPy judge module once or N times, retrying ``AdapterParseError`` per call."""

    def _call_with_retries(rollout_id: int | None) -> JudgeResult:
        last_exc: Exception | None = None
        for attempt in range(JUDGE_PARSE_RETRIES + 1):
            try:
                if rollout_id is None and attempt == 0:
                    return evaluator(**kwargs)
                effective_rollout_id = (rollout_id if rollout_id is not None else 0) + attempt
                effective_temperature = temperature if rollout_id is not None else max(temperature, 1.0)
                return evaluator(
                    **kwargs,
                    config={"rollout_id": effective_rollout_id, "temperature": effective_temperature},
                )
            except AdapterParseError as e:
                last_exc = e
                lm_response = getattr(e, "lm_response", "") or ""
                head = lm_response[:300].replace("\n", "\\n")
                tail = lm_response[-300:].replace("\n", "\\n") if len(lm_response) > 300 else ""
                logger.warning(
                    "Judge parse failure (attempt %d/%d, rollout_id=%s, finish_reason=%s, "
                    "lm_response_len=%d, lm_response_head=%r, lm_response_tail=%r): %s",
                    attempt + 1,
                    JUDGE_PARSE_RETRIES + 1,
                    rollout_id,
                    _peek_last_finish_reason(),
                    len(lm_response),
                    head,
                    tail,
                    str(e).splitlines()[0][:200],
                )
            except JUDGE_API_ERRORS as e:
                logger.warning(
                    "Judge LM API error after dspy retries (rollout_id=%s): %s: %s",
                    rollout_id,
                    type(e).__name__,
                    str(e).splitlines()[0][:200],
                )
                raise
        assert last_exc is not None
        raise last_exc

    if n <= 1:
        return [_call_with_retries(rollout_id=None)]

    results: list[JudgeResult] = []
    last_exc: Exception | None = None
    for shot in range(n):
        try:
            results.append(_call_with_retries(rollout_id=shot * (JUDGE_PARSE_RETRIES + 1)))
        except JUDGE_ERRORS as e:
            if isinstance(e, JUDGE_FATAL_ERRORS):
                raise
            last_exc = e
    if not results:
        assert last_exc is not None
        raise last_exc
    if len(results) < n:
        logger.warning(
            "Multi-shot judge call had partial failures: %d/%d rollouts succeeded. "
            "Aggregating over surviving rollouts.",
            len(results),
            n,
        )
    return results


class CatchJudgeFailure:
    """Context manager that turns a judge failure inside the block into a failed row instead of an ended run.

    A judge error after all retries (``JUDGE_ERRORS``) is logged and suppressed, and ``failed`` is set. Fatal judge
    errors (``JUDGE_FATAL_ERRORS``) and every other exception propagate::

        with CatchJudgeFailure(f"row {i}") as caught:
            ...  # judge calls
        if caught.failed:
            ...  # record the row as judge_failed
    """

    def __init__(self, row: str) -> None:
        """Name the row in the log message (e.g. ``"row 3"`` or the question)."""
        self.row = row
        self.failed = False

    def __enter__(self) -> Self:
        """Enter the block."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> bool:
        """Suppress a non-fatal judge error and mark the row as failed."""
        if not isinstance(exc, JUDGE_ERRORS) or isinstance(exc, JUDGE_FATAL_ERRORS):
            return False
        self.failed = True
        message = str(exc).splitlines()[0][:200] if str(exc) else ""
        logger.error(
            "Judge failed on %s after retries (%s): %s. Recording judge_failed.", self.row, type(exc).__name__, message
        )
        return True


def judge_rows[T](
    fn: Callable[[int], T], indices: list[int], num_threads: int, desc: str
) -> tuple[list[T | None], set[int]]:
    """Apply a per-row judge function to every index, in parallel threads if num_threads > 1.

    Returns the results in input order and the indices whose judge failed (see ``CatchJudgeFailure``); a failed
    row's result is None. Any other exception is re-raised and queued rows are cancelled instead of being worked off.
    """
    failed: set[int] = set()

    def guarded(i: int) -> T | None:
        with CatchJudgeFailure(f"row {i}") as caught:
            return fn(i)
        if caught.failed:
            failed.add(i)
        return None

    if num_threads <= 1:
        return [guarded(i) for i in tqdm(indices, desc=desc)], failed
    executor = ThreadPoolExecutor(max_workers=num_threads)
    try:
        results = list(tqdm(executor.map(guarded, indices), total=len(indices), desc=desc))
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown()
    return results, failed
