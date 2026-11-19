from types import SimpleNamespace

import pytest
from dspy.utils.exceptions import AdapterParseError, LMRateLimitError

from lmbench.config.config import CONFIG
from lmbench.metrics.hallucination import (
    _PUBLISHED_SCORE_KEY,
    COMPOSITE_SCORE,
    HallucinationMetric,
)
from lmbench.metrics.judge.claims import ClaimExtractionSignature


def test_every_scored_sub_score_has_a_published_name():
    """Scoring names and published names are declared apart; they must stay in step."""
    scored = {key for keys in HallucinationMetric.APPLICABLE_SCORES.values() for key in keys}
    assert set(_PUBLISHED_SCORE_KEY) == scored | {COMPOSITE_SCORE}


def test_published_score_names_are_the_ones_the_run_archive_uses():
    """These names are the output columns of every archived run; renaming one breaks comparability."""
    assert set(_PUBLISHED_SCORE_KEY.values()) == {
        "hallucination_benchmark_score",
        "status_accuracy",
        "factual_correctness",
        "reference_f1",
        "context_groundedness",
        "refusal_quality",
        "knowledge_leakage_avoidance",
        "conflict_handling",
    }


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


def _build_metric(dataset: DummyDataset, weights: dict[str, dict[str, float]], judge_n: int = 1) -> HallucinationMetric:
    metric = HallucinationMetric.__new__(HallucinationMetric)
    metric.dataset = dataset
    metric.selected_indices = [2]
    metric.config = SimpleNamespace(
        judge_n=judge_n,
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


def test_weight_keys_outside_the_question_type_are_rejected():
    """A weight the branch never scores would silently drag every composite of that type down."""
    metric = _build_metric(DummyDataset(), {"answerable": {"refusal_quality": 0.4}})
    with pytest.raises(ValueError, match="not scored for ANSWERABLE"):
        metric._check_weight_keys()


def test_the_retired_weight_name_is_rejected_rather_than_silently_ignored():
    metric = _build_metric(DummyDataset(), {"unanswerable": {"knowledge_leakage": 0.3}})
    with pytest.raises(ValueError, match="knowledge_leakage_avoidance"):
        metric._check_weight_keys()


def test_default_weights_name_only_sub_scores_their_question_type_scores():
    metric = _build_metric(DummyDataset(), {})
    metric.config = CONFIG.hallucination
    metric._check_weight_keys()


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
                "knowledge_leakage_avoidance": 0.3,
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
    return AdapterParseError(adapter_name="TestAdapter", signature=ClaimExtractionSignature, lm_response="<malformed>")


class _FailingEvaluator:
    """Sub-evaluator stub that always fails the way an exhausted judge does."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def __call__(self, **kwargs) -> SimpleNamespace:  # noqa: ARG002
        raise self.error


def test_item_statuses_leave_inapplicable_dimensions_unset(monkeypatch):
    """A CONFLICTING row is not scored on factual correctness, so that dimension has no status."""
    dataset = DummyDataset(question_types=["ANSWERABLE", "ANSWERABLE", "CONFLICTING"])
    metric = _build_metric(
        dataset,
        weights={
            "answerable": {"status_correctness": 1.0},
            "unanswerable": {"status_correctness": 1.0},
            "conflicting": {"status_correctness": 0.5, "conflict_detection_quality": 0.5},
        },
    )
    _patch_parser(monkeypatch, "CONFLICTING")

    metric.evaluate(["model-response"], ["ga-c"])

    statuses = metric.item_statuses()
    assert statuses is not None
    assert statuses["status_accuracy"] == ["ok"]
    assert statuses["conflict_handling"] == ["ok"]
    assert statuses["reference_f1"] == ["ok"]
    assert statuses["hallucination_benchmark_score"] == ["ok"]
    # Dimensions this question type does not score are NaN by design, not judge failures.
    assert statuses["factual_correctness"] == [None]
    assert statuses["context_groundedness"] == [None]
    assert statuses["refusal_quality"] == [None]
    assert statuses["knowledge_leakage_avoidance"] == [None]


@pytest.mark.parametrize(
    "error",
    [_parse_error(), LMRateLimitError("429")],
    ids=["unparseable answer", "API error after retries"],
)
def test_item_statuses_report_a_judge_failure(monkeypatch, error):
    dataset = DummyDataset(question_types=["ANSWERABLE", "ANSWERABLE", "ANSWERABLE"])
    metric = _build_metric(
        dataset,
        weights={"answerable": {"status_correctness": 1.0}, "unanswerable": {}, "conflicting": {}},
    )
    metric._factual_correctness = _FailingEvaluator(error)
    _patch_parser(monkeypatch, "ANSWERABLE")

    metric.evaluate(["model-response"], ["ga-c"])

    statuses = metric.item_statuses()
    assert statuses is not None
    # The row aborts before any sub-score lands, including the ones evaluated earlier in the branch.
    assert statuses["factual_correctness"] == ["judge_failed"]
    assert statuses["hallucination_benchmark_score"] == ["judge_failed"]
    # Not applicable to ANSWERABLE, so still unset rather than failed.
    assert statuses["refusal_quality"] == [None]
