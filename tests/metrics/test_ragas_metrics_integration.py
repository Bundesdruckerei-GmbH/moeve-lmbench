"""
Integration tests for Ragas metrics using real LLM calls.

These tests will only run if the necessary Azure OpenAI environment variables are
available.  Otherwise they are skipped automatically.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from lmbench.dataset.dataframe import DataframeDataset, DataframeDatasetConfig
from lmbench.metrics.ragas_metrics import RagasComparisonMetrics, RagasComparisonMetricsV2, RagasQAMetrics


class _DummyDataframeDataset(DataframeDataset):
    """
    Minimal `DataframeDataset` implementation that avoids any file-system
    interaction while still satisfying the type check inside
    `RagasQAMetrics.evaluate`.
    """

    def __init__(self, data: pd.DataFrame, config: DataframeDatasetConfig):
        # Intentionally *not* calling super().__init__ to skip I/O
        self.dataset = data
        self.config = config


def _azure_openai_available() -> bool:
    "Return True iff all variables required for a real Azure OpenAI call exist."
    required_env_vars = ("AZURE_OPENAI_KEY",)
    return all(os.getenv(v) for v in required_env_vars)


pytestmark = pytest.mark.skipif(
    not _azure_openai_available(),
    reason="Azure OpenAI environment variables not set – skipping integration tests.",
)


@pytest.fixture(autouse=True)
def _isolated_ragas_cache(tmp_path, monkeypatch):
    """Point the ragas judge cache at a fresh tmp dir per test.

    Stops a wrong cached judge response in cache/cache.db from making this
    integration test deterministically fail across runs.
    """
    monkeypatch.setattr("lmbench.metrics.ragas_metrics.CACHE_FOLDER", tmp_path)


def test_ragas_comparison_metrics_integration():
    metric = RagasComparisonMetrics()
    model_output = [
        "Berlin ist die Hauptstadt von Deutschland.",
        "Paris ist die Hauptstadt von Deutschland.",
    ]
    labels = [
        "Berlin ist die Hauptstadt von Deutschland.",
        "Paris ist die Hauptstadt von Frankreich.",
    ]

    result = metric.evaluate(model_output, labels)

    assert result["factual_correctness"][0] == pytest.approx(1.0, rel=0.01)
    # The LLM may split the incorrect claim into 1 or 2 sub-claims, yielding 0.0 or 0.5.
    # We just verify it's recognized as mostly incorrect.
    assert result["factual_correctness"][1] <= 0.5


@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_ragas_qa_metrics_integration():
    df = pd.DataFrame(
        {
            "question_column": [
                "Was ist die Hauptstadt von Deutschland?",
                "Was ist die Hauptstadt von Frankreich?",
            ],
            "context_column": [
                "Berlin ist die Hauptstadt von Deutschland.",
                "Paris ist die Hauptstadt von Frankreich.",
            ],
            "target_column": [
                "Berlin ist die Hauptstadt von Deutschland.",
                "Paris ist die Hauptstadt von Frankreich.",
            ],
        }
    )

    config = DataframeDatasetConfig(
        system_prompt="system_prompt.txt",
        user_prompt="user_prompt.txt",
        target_column="target_column",
        shrink_column="context_column",
        context_column="context_column",
        question_column="question_column",
    )

    dataset = _DummyDataframeDataset(df, config)
    metric = RagasQAMetrics(dataset)

    model_output = [
        "Berlin ist die Hauptstadt von Deutschland.",
        "Berlin ist die Hauptstadt von Frankreich.",
    ]
    labels = [
        "Berlin ist die Hauptstadt von Deutschland.",
        "Paris ist die Hauptstadt von Frankreich.",
    ]

    result = metric.evaluate(model_output, labels)

    assert result["faithfulness"][0] == pytest.approx(1.0, rel=0.01)
    assert result["faithfulness"][1] == pytest.approx(0.0, rel=0.01)

    assert result["noise_sensitivity_relevant"][0] == pytest.approx(0.0, rel=0.01)
    assert result["noise_sensitivity_relevant"][1] == pytest.approx(0.0, rel=0.01)


@pytest.mark.skip(reason="V2 API is experimental and produces occasional failures")
def test_ragas_comparison_metrics_v2_integration():
    """Test the V2 implementation using llm_factory and @experiment decorator."""
    metric = RagasComparisonMetricsV2()
    model_output = [
        "Berlin ist die Hauptstadt von Deutschland.",
        "Paris ist die Hauptstadt von Deutschland.",
    ]
    labels = [
        "Berlin ist die Hauptstadt von Deutschland.",
        "Paris ist die Hauptstadt von Frankreich.",
    ]

    result = metric.evaluate(model_output, labels)

    assert "factual_correctness" in result
    assert len(result["factual_correctness"]) == 2
    assert result["factual_correctness"][0] == pytest.approx(1.0, rel=0.01)
    # The LLM may split the incorrect claim into 1 or 2 sub-claims, yielding 0.0 or 0.5.
    # We just verify it's recognized as mostly incorrect.
    assert result["factual_correctness"][1] <= 0.5
