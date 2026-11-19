"""Tests of the shared LLM-judge building blocks (lmbench.metrics.judge)."""

from types import SimpleNamespace

import pytest
from dspy.utils.exceptions import AdapterParseError, LMAuthError, LMRateLimitError

from lmbench.metrics.judge.calls import JUDGE_PARSE_RETRIES, CatchJudgeFailure, call_judge_n, judge_rows
from lmbench.metrics.judge.claims import ClaimExtractionSignature
from lmbench.metrics.judge.lm import HarmonyUnwrappingJSONAdapter


def _parse_error() -> AdapterParseError:
    return AdapterParseError(
        adapter_name="TestAdapter",
        signature=ClaimExtractionSignature,
        lm_response="<malformed>",
    )


class _ScriptedEvaluator:
    """Mock dspy module that records calls and follows a scripted failure pattern."""

    def __init__(self, failure_pattern: list[bool] | None = None, default_failure: bool = False) -> None:
        self.calls: list[dict] = []
        self.failure_pattern = failure_pattern or []
        self.default_failure = default_failure

    def __call__(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        idx = len(self.calls) - 1
        should_fail = self.failure_pattern[idx] if idx < len(self.failure_pattern) else self.default_failure
        if should_fail:
            raise _parse_error()
        return SimpleNamespace(score=1.0, verdicts=[True])


def test_call_judge_n_single_shot_success_first_try():
    """n=1, first attempt succeeds → returns one result, no config override passed."""
    evaluator = _ScriptedEvaluator()
    results = call_judge_n(evaluator, n=1, temperature=0.0, x="hello")
    assert len(results) == 1
    assert len(evaluator.calls) == 1
    # First-attempt single-shot does NOT pass a per-call config override.
    assert "config" not in evaluator.calls[0]
    assert evaluator.calls[0]["x"] == "hello"


def test_call_judge_n_single_shot_retry_then_success():
    """n=1, first two attempts fail then third succeeds → returns one result, three calls."""
    evaluator = _ScriptedEvaluator(failure_pattern=[True, True, False])
    results = call_judge_n(evaluator, n=1, temperature=0.0, x="hello")
    assert len(results) == 1
    assert len(evaluator.calls) == 3
    # First call: no override.
    assert "config" not in evaluator.calls[0]
    # Retries: distinct rollout_ids, temperature clamped to ≥1.0 (since rollout_id is None mode).
    assert evaluator.calls[1]["config"] == {"rollout_id": 1, "temperature": 1.0}
    assert evaluator.calls[2]["config"] == {"rollout_id": 2, "temperature": 1.0}


def test_call_judge_n_single_shot_all_attempts_fail_reraises():
    """n=1, all attempts fail → original AdapterParseError propagates."""
    evaluator = _ScriptedEvaluator(default_failure=True)
    with pytest.raises(AdapterParseError):
        call_judge_n(evaluator, n=1, temperature=0.0, x="hello")
    assert len(evaluator.calls) == JUDGE_PARSE_RETRIES + 1


def test_call_judge_n_multi_shot_all_succeed():
    """n=3, all shots succeed first try → 3 results, spaced rollout_ids, no overlaps."""
    evaluator = _ScriptedEvaluator()
    results = call_judge_n(evaluator, n=3, temperature=1.0, x="hello")
    assert len(results) == 3
    assert len(evaluator.calls) == 3
    rollout_ids = [call["config"]["rollout_id"] for call in evaluator.calls]
    # Spacing is shot * (JUDGE_PARSE_RETRIES + 1); with JUDGE_PARSE_RETRIES=2 → 0, 3, 6.
    assert rollout_ids == [shot * (JUDGE_PARSE_RETRIES + 1) for shot in range(3)]


def test_call_judge_n_multi_shot_passes_temperature_through_unclamped():
    """Multi-shot uses the caller's temperature verbatim (no max() clamp)."""
    evaluator = _ScriptedEvaluator()
    call_judge_n(evaluator, n=3, temperature=0.5, x="hello")
    for call in evaluator.calls:
        assert call["config"]["temperature"] == 0.5


def test_call_judge_n_multi_shot_partial_failure_drops_failed_shots():
    """n=3, middle shot fails all 3 attempts → returns 2 surviving results."""
    # Pattern: shot 0 success (1 call), shot 1 fails all 3 attempts (3 calls), shot 2 success (1 call).
    evaluator = _ScriptedEvaluator(failure_pattern=[False, True, True, True, False])
    results = call_judge_n(evaluator, n=3, temperature=1.0, x="hello")
    assert len(results) == 2  # only the two surviving shots
    assert len(evaluator.calls) == 5


def test_call_judge_n_multi_shot_all_shots_fail_reraises():
    """n=3, every shot fails all attempts → re-raises after exhausting all budgets."""
    evaluator = _ScriptedEvaluator(default_failure=True)
    with pytest.raises(AdapterParseError):
        call_judge_n(evaluator, n=3, temperature=1.0, x="hello")
    # 3 shots × (JUDGE_PARSE_RETRIES + 1) attempts each.
    assert len(evaluator.calls) == 3 * (JUDGE_PARSE_RETRIES + 1)


def test_call_judge_n_multi_shot_survives_an_api_error_after_retries():
    """dspy 3 raises its own LMError types for API failures; one failed shot drops only that shot."""
    calls = []

    def evaluator(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise LMRateLimitError("429")
        return SimpleNamespace(score=1.0)

    results = call_judge_n(evaluator, n=3, temperature=1.0, x="hello")
    assert len(results) == 2
    assert len(calls) == 3  # API errors are retried by the LM itself, not again here


def test_call_judge_n_auth_error_is_not_swallowed():
    """A wrong key would fail every call, so it ends the run even in multi-shot mode."""

    def evaluator(**_):
        raise LMAuthError("401")

    with pytest.raises(LMAuthError):
        call_judge_n(evaluator, n=3, temperature=1.0, x="hello")


def test_call_judge_n_rollout_ids_distinct_across_shots_and_retries():
    """Spaced rollout_id allocation guarantees no cache-key collisions across shots."""
    # Each shot fails twice then succeeds. JUDGE_PARSE_RETRIES=2 → 3 attempts per shot.
    evaluator = _ScriptedEvaluator(failure_pattern=[True, True, False] * 3)
    results = call_judge_n(evaluator, n=3, temperature=1.0, x="hello")
    assert len(results) == 3
    assert len(evaluator.calls) == 9
    rollout_ids = [call["config"]["rollout_id"] for call in evaluator.calls]
    # Shot 0: 0, 1, 2  | Shot 1: 3, 4, 5  | Shot 2: 6, 7, 8
    assert rollout_ids == list(range(9))
    assert len(set(rollout_ids)) == 9  # no collisions


def test_harmony_unwrap_strips_final_envelope():
    wrapped = '{"final": "{\\"reasoning\\": \\"all good\\", \\"verdicts\\": [true, false]}"}'
    expected_inner = '{"reasoning": "all good", "verdicts": [true, false]}'
    assert HarmonyUnwrappingJSONAdapter._unwrap_harmony(wrapped) == expected_inner


def test_harmony_unwrap_passes_through_bare_json():
    """Normal dspy output (no envelope) must be returned unchanged."""
    bare = '{"reasoning": "all good", "verdicts": [true, false]}'
    assert HarmonyUnwrappingJSONAdapter._unwrap_harmony(bare) == bare


def test_harmony_unwrap_passes_through_empty_string():
    assert HarmonyUnwrappingJSONAdapter._unwrap_harmony("") == ""


def test_harmony_unwrap_passes_through_malformed_json():
    """Non-JSON or truncated responses must not raise — adapter's parse() will report it."""
    assert HarmonyUnwrappingJSONAdapter._unwrap_harmony("not json at all {") == "not json at all {"


def test_harmony_unwrap_strips_final_envelope_dict_variant():
    """Foundry also emits {"final": {<object>}} — re-serialize the inner object to JSON."""
    wrapped = '{"final": {"verdicts": [true, false, true]}}'
    expected_inner = '{"verdicts": [true, false, true]}'
    assert HarmonyUnwrappingJSONAdapter._unwrap_harmony(wrapped) == expected_inner


def test_harmony_unwrap_ignores_envelopes_with_extra_keys():
    """Only the strict single-key {final: ...} envelope is unwrapped."""
    other = '{"final": "x", "extra": "y"}'
    assert HarmonyUnwrappingJSONAdapter._unwrap_harmony(other) == other


@pytest.mark.parametrize("error", [_parse_error(), LMRateLimitError("429")], ids=["parse", "api"])
def test_catch_judge_failure_suppresses_a_judge_error_and_marks_the_row(error):
    with CatchJudgeFailure("row 0") as caught:
        raise error
    assert caught.failed


@pytest.mark.parametrize("error", [LMAuthError("401"), ValueError("bug")], ids=["fatal", "other"])
def test_catch_judge_failure_lets_fatal_and_other_errors_through(error):
    with pytest.raises(type(error)), CatchJudgeFailure("row 0"):
        raise error


def test_catch_judge_failure_without_error_is_not_failed():
    with CatchJudgeFailure("row 0") as caught:
        pass
    assert not caught.failed


@pytest.mark.parametrize("num_threads", [1, 4])
def test_judge_rows_keeps_order_and_reports_failed_rows(num_threads):
    def judge(i: int) -> int:
        if i == 2:
            raise LMRateLimitError("429")
        return i * 10

    results, failed = judge_rows(judge, [0, 1, 2, 3], num_threads, "test")

    assert results == [0, 10, None, 30]
    assert failed == {2}


@pytest.mark.parametrize("num_threads", [1, 4])
def test_judge_rows_reraises_fatal_errors(num_threads):
    def judge(_: int) -> int:
        raise LMAuthError("401")

    with pytest.raises(LMAuthError):
        judge_rows(judge, [0, 1], num_threads, "test")
