"""
Integration test for HallucinationMetric using a real LLM judge.

Skipped automatically when AZURE_AI_FOUNDRY_KEY (the env var consumed by
config.hallucination_judge) is not set.
"""

from __future__ import annotations

import os

import pytest


class _StubDataset:
    """Minimal Dataset stand-in implementing the `get_column` interface used by HallucinationMetric."""

    def __init__(self, columns: dict[str, list]) -> None:
        self._columns = columns

    def get_column(self, name: str) -> list:
        return self._columns[name]


def _hallucination_judge_available() -> bool:
    return bool(os.getenv("AZURE_AI_FOUNDRY_KEY"))


pytestmark = pytest.mark.skipif(
    not _hallucination_judge_available(),
    reason="AZURE_AI_FOUNDRY_KEY not set — skipping hallucination integration test.",
)


@pytest.fixture(autouse=True)
def _isolated_dspy_settings():
    """Snapshot and restore dspy.settings around each test.

    ``HallucinationMetric.__init__`` calls ``dspy.configure(lm=...)``, which
    mutates process-global state. Without isolation the LM stays configured
    after this test and leaks into any subsequent dspy-using test in the same
    pytest session. ``dspy.settings.context()`` alone is not enough because
    ``dspy.configure`` writes to ``main_thread_config`` directly, bypassing
    the thread-local override layer that ``context()`` manages.
    """
    import dspy

    saved_lm = dspy.settings.lm
    saved_adapter = dspy.settings.adapter
    try:
        yield
    finally:
        dspy.configure(lm=saved_lm, adapter=saved_adapter)


@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_hallucination_metric_answerable_end_to_end():
    """A well-formed ANSWERABLE response with correct content should score high end-to-end."""
    from lmbench.metrics.hallucination import HallucinationMetric

    dataset = _StubDataset(
        {
            "question": ["Was ist die Hauptstadt von Deutschland?"],
            "question_type": ["ANSWERABLE"],
            "first_document": ["Berlin ist die Hauptstadt von Deutschland."],
            "second_document": [""],
            "third_document": [""],
            "gold_answer": ["Berlin ist die Hauptstadt von Deutschland."],
            "references": [[1]],
        }
    )
    metric = HallucinationMetric(dataset=dataset, selected_indices=[0])  # type: ignore[arg-type]

    model_output = ["STATUS: ANSWERABLE\nDOKUMENTE: 1\nANTWORT: Berlin ist die Hauptstadt von Deutschland."]
    result = metric.evaluate(model_output, ["Berlin ist die Hauptstadt von Deutschland."])

    assert set(result.keys()) >= {
        "hallucination_benchmark_score",
        "status_accuracy",
        "factual_correctness",
        "reference_f1",
        "context_groundedness",
    }
    assert len(result["hallucination_benchmark_score"]) == 1

    # STATUS matched exactly -> perfect status accuracy.
    assert result["status_accuracy"][0] == 1.0
    # Document reference F1 is pure Python (no LLM): predicted [1] vs gold [1] -> 1.0.
    assert result["reference_f1"][0] == 1.0
    # LLM-judged sub-scores: should all be high for this trivially correct response.
    assert result["factual_correctness"][0] >= 0.7
    assert result["context_groundedness"][0] >= 0.7
    # Weighted aggregate must reflect that.
    assert result["hallucination_benchmark_score"][0] >= 0.7
