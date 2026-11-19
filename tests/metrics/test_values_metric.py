from pathlib import Path

import pytest

from lmbench.config.config import RESOURCES_FOLDER
from lmbench.metrics.values import ValuesMetric, ValuesMetricBERT

_BERT_MODEL_PATH = Path(RESOURCES_FOLDER) / "hf_model" / "GottBERT_base_last_2025-10-24"


class DummyService:
    def invoke(self, messages, response_format):  # noqa: ARG002
        return {"standpunkt": "pro", "begründung": "just"}


class StanceService:
    """Judge stub returning a fixed stance per call, in order."""

    def __init__(self, stances: list[str]) -> None:
        self.stances = list(stances)

    def invoke(self, messages, response_format):  # noqa: ARG002
        return {"standpunkt": self.stances.pop(0), "begründung": "just"}


class DummyDataset:
    def __init__(self, framings: list[str] | None = None) -> None:
        self.framings = framings

    def get_column(self, _name: str) -> list:
        # Provide a dummy framing sequence (empty strings) – length must match model_output in tests
        return self.framings if self.framings is not None else ["f"] * 4


class DummySegmenter:
    def segment(self, text):  # noqa: ARG002
        return ["This is sentence 1.", "This is sentence 2."]


def test_values_metric_uniform_pro(monkeypatch):
    vm = ValuesMetric(dataset=DummyDataset())
    monkeypatch.setattr(vm, "llm_service", DummyService())
    model_output = ["out1", "out2", "out3", "out4"]
    target_values = ["v1", "v1", "v2", "v2"]
    result = vm.evaluate(model_output, target_values)
    assert result == {"stance_proportion_v1_f": 1.0, "stance_proportion_v2_f": 1.0}


def test_values_metric_uneven_group_sizes(monkeypatch):
    """A group with fewer generations must be normalized by its own size, not the average size."""
    vm = ValuesMetric(dataset=DummyDataset(framings=["f", "f", "f"]))
    monkeypatch.setattr(vm, "llm_service", StanceService(["pro", "kontra", "pro"]))
    result = vm.evaluate(["out1", "out2", "out3"], ["v1", "v1", "v2"])
    assert result == {"stance_proportion_v1_f": 0.5, "stance_proportion_v2_f": 1.0}


def test_values_metric_item_scores_average_to_the_reported_proportion(monkeypatch):
    vm = ValuesMetric(dataset=DummyDataset(framings=["f", "f", "f"]))
    monkeypatch.setattr(vm, "llm_service", StanceService(["pro", "kontra", "pro"]))
    result = vm.evaluate(["out1", "out2", "out3"], ["v1", "v1", "v2"])

    item_scores = vm.item_scores()
    assert item_scores is not None
    for key, proportion in result.items():
        present = [score for score in item_scores[key] if score is not None]
        assert len(item_scores[key]) == 3
        assert sum(present) / len(present) == proportion
    assert item_scores["stance_proportion_v1_f"] == [1.0, 0.0, None]
    assert item_scores["stance_proportion_v2_f"] == [None, None, 1.0]


def test_values_metric_bert_uniform_pro(monkeypatch):
    if not _BERT_MODEL_PATH.is_dir():
        pytest.skip(f"GottBERT model not found at {_BERT_MODEL_PATH}")
    vm = ValuesMetricBERT(dataset=DummyDataset())
    monkeypatch.setattr(vm, "sentence_segmenter", DummySegmenter())
    model_output = ["out1", "out2", "out3", "out4"]
    target_values = ["v1", "v1", "v2", "v2"]
    result = vm.evaluate(model_output, target_values)
    assert result == {"stance_proportion_BERT_v1_f": 0.5, "stance_proportion_BERT_v2_f": 0.5}
