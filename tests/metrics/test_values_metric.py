from pathlib import Path

import pytest

from lmbench.config.config import RESOURCES_FOLDER
from lmbench.metrics.values import ValuesMetric, ValuesMetricBERT

_BERT_MODEL_PATH = Path(RESOURCES_FOLDER) / "hf_model" / "GottBERT_base_last_2025-10-24"


class DummyService:
    def invoke(self, messages, response_format):  # noqa: ARG002
        return {"standpunkt": "pro", "begründung": "just"}


class DummyDataset:
    def get_column(self, _name: str) -> list:
        # Provide a dummy framing sequence (empty strings) – length must match model_output in tests
        return ["f"] * 4


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


def test_values_metric_bert_uniform_pro(monkeypatch):
    if not _BERT_MODEL_PATH.is_dir():
        pytest.skip(f"GottBERT model not found at {_BERT_MODEL_PATH}")
    vm = ValuesMetricBERT(dataset=DummyDataset())
    monkeypatch.setattr(vm, "sentence_segmenter", DummySegmenter())
    model_output = ["out1", "out2", "out3", "out4"]
    target_values = ["v1", "v1", "v2", "v2"]
    result = vm.evaluate(model_output, target_values)
    assert result == {"stance_proportion_BERT_v1_f": 0.5, "stance_proportion_BERT_v2_f": 0.5}
