from types import SimpleNamespace

import pytest
from dspy.utils.exceptions import AdapterParseError

from lmbench.metrics.hallucination import (
    JUDGE_PARSE_RETRIES,
    ClaimExtractionSignature,
    HallucinationMetric,
    HarmonyUnwrappingJSONAdapter,
    call_judge_n,
)


class DummyDataset:
    def __init__(self, question_types: list[str] | None = None) -> None:
        self.columns = {
            "question": ["wrong-row", "also-wrong", "selected-question"],
            "question_type": question_types or ["ANSWERABLE", "ANSWERABLE", "ANSWERABLE"],
            "first_document": ["doc-a", "doc-b", "doc-c"],
            "second_document": ["", "", ""],
            "third_document": ["", "", ""],
            "gold_answer": ["ga-a", "ga-b", "ga-c"],
            "references": [[1], [2], [3]],
        }

    def get_column(self, name: str) -> list:
        return self.columns[name]


class RecordingEvaluator:
    """Sub-evaluator stub that records every kwargs-only call it receives."""

    def __init__(self, score: float = 1.0) -> None:
        self.score = score
        self.calls: list[dict] = []

    def __call__(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(score=self.score, judge_score_variance=0.0)


def _build_metric(dataset: DummyDataset, weights: dict[str, dict[str, float]]) -> HallucinationMetric:
    metric = HallucinationMetric.__new__(HallucinationMetric)
    metric.dataset = dataset
    metric.selected_indices = [2]
    metric.config = SimpleNamespace(
        judge_n=1,
        answerable_weights=weights.get("answerable", {}),
        unanswerable_weights=weights.get("unanswerable", {}),
        conflicting_weights=weights.get("conflicting", {}),
    )
    metric._status_correctness = RecordingEvaluator()
    metric._document_accuracy = RecordingEvaluator()
    metric._factual_correctness = RecordingEvaluator()
    metric._context_groundedness = RecordingEvaluator()
    metric._refusal_quality = RecordingEvaluator()
    metric._knowledge_leakage = RecordingEvaluator()
    metric._conflict_detection_quality = RecordingEvaluator()
    return metric


def _patch_parser(monkeypatch, status: str) -> None:
    monkeypatch.setattr(
        "lmbench.metrics.hallucination.parse_hallucination_output",
        lambda output: SimpleNamespace(
            status=SimpleNamespace(value=status),
            documents=[3],
            response_text=output,
        ),
    )


def test_hallucination_metric_uses_selected_indices(monkeypatch):
    dataset = DummyDataset()  # default ANSWERABLE rows
    metric = _build_metric(
        dataset,
        weights={
            "answerable": {"status_correctness": 0.5, "factual_correctness": 0.5},
            "unanswerable": {},
            "conflicting": {},
        },
    )
    _patch_parser(monkeypatch, "ANSWERABLE")

    result = metric.evaluate(["model-response"], ["ga-c"])

    # selected_indices=[2] → only the third row ("selected-question") should be evaluated.
    assert [call["question"] for call in metric._factual_correctness.calls] == ["selected-question"]
    assert result["hallucination_benchmark_score"] == [1.0]


def test_hallucination_metric_unanswerable_routing(monkeypatch):
    dataset = DummyDataset(question_types=["ANSWERABLE", "ANSWERABLE", "UNANSWERABLE"])
    metric = _build_metric(
        dataset,
        weights={
            "answerable": {"status_correctness": 1.0},
            "unanswerable": {
                "status_correctness": 0.3,
                "refusal_quality": 0.4,
                "knowledge_leakage": 0.3,
            },
            "conflicting": {"status_correctness": 1.0},
        },
    )
    _patch_parser(monkeypatch, "UNANSWERABLE")

    result = metric.evaluate(["model-response"], ["ga-c"])

    # Routed to UNANSWERABLE-only sub-evaluators.
    assert len(metric._refusal_quality.calls) == 1
    assert len(metric._knowledge_leakage.calls) == 1
    # ANSWERABLE/CONFLICTING-only sub-evaluators are skipped.
    assert metric._factual_correctness.calls == []
    assert metric._context_groundedness.calls == []
    assert metric._conflict_detection_quality.calls == []
    # _document_accuracy is only used for ANSWERABLE and CONFLICTING.
    assert metric._document_accuracy.calls == []
    # All sub-scores = 1.0; weighted sum of unanswerable_weights = 1.0.
    assert result["hallucination_benchmark_score"] == [1.0]


def test_hallucination_metric_conflicting_routing(monkeypatch):
    dataset = DummyDataset(question_types=["ANSWERABLE", "ANSWERABLE", "CONFLICTING"])
    metric = _build_metric(
        dataset,
        weights={
            "answerable": {"status_correctness": 1.0},
            "unanswerable": {"status_correctness": 1.0},
            "conflicting": {
                "status_correctness": 0.3,
                "conflict_detection_quality": 0.4,
                "document_f1": 0.3,
            },
        },
    )
    _patch_parser(monkeypatch, "CONFLICTING")

    result = metric.evaluate(["model-response"], ["ga-c"])

    assert len(metric._conflict_detection_quality.calls) == 1
    assert len(metric._document_accuracy.calls) == 1
    # UNANSWERABLE and pure-ANSWERABLE sub-evaluators are skipped.
    assert metric._refusal_quality.calls == []
    assert metric._knowledge_leakage.calls == []
    assert metric._factual_correctness.calls == []
    assert metric._context_groundedness.calls == []
    assert result["hallucination_benchmark_score"] == [1.0]


def test_hallucination_metric_parse_error_does_not_crash():
    """Garbage model output produces PARSE_ERROR; the metric should still return a finite row."""
    dataset = DummyDataset(question_types=["ANSWERABLE", "ANSWERABLE", "ANSWERABLE"])
    metric = _build_metric(
        dataset,
        weights={
            "answerable": {
                "status_correctness": 0.2,
                "factual_correctness": 0.3,
                "document_f1": 0.3,
                "context_groundedness": 0.2,
            },
            "unanswerable": {},
            "conflicting": {},
        },
    )
    # Status mismatch: PARSE_ERROR != ANSWERABLE -> score 0.
    metric._status_correctness = RecordingEvaluator(score=0.0)

    # No parser monkeypatch: real parse_hallucination_output returns PARSE_ERROR for garbage.
    result = metric.evaluate(["this is not a structured response"], ["ga-c"])

    assert len(result["hallucination_benchmark_score"]) == 1
    # status_correctness=0 contributes 0; remaining 0.8 of weights from sub-evaluators (all return 1.0).
    assert result["hallucination_benchmark_score"][0] == 0.8
    assert metric._status_correctness.calls[0]["llm_status"] == "PARSE_ERROR"


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
