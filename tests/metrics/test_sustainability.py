from unittest.mock import MagicMock

import pytest

from lmbench.metrics.sustainability import SustainabilityMetrics
from lmbench.models.data_models import LLMOutput
from lmbench.tokenizer.simple import SimpleTokenizer


class DummyModel:
    """A dummy LLM model with zero parameters and a SimpleTokenizer."""

    def __init__(self):
        self.active_parameters = 0
        self.total_parameters = 0
        self.tokenizer = SimpleTokenizer(self)


def test_non_reasoning():
    dummy_model = DummyModel()
    metric = SustainabilityMetrics(model=dummy_model)
    model_output = ["Das ist ein Satz mit 7 Wörtern.", "Das ist jetzt ein anderer Satz der 10 Wörter enthält."]
    expected = {
        "full_tokens": [38.0, 50.0],  # Simple-Tokenizer: Tokens = Number of words * 4 + 10 tokens overhead
        "full_words": [7.0, 10.0],  # 7 Wörter im ersten Satz, 10 Wörter im zweiten Satz
        "answer_words": [7.0, 10.0],  # 7 Wörter im ersten Satz, 10 Wörter im zweiten Satz
        "reasoning_words": [0.0, 0.0],  # Keine Reasoning Tokens
        "energy": [0.0, 0.0],
        "gwp_de": [0.0, 0.0],
        "gwp_eu": [0.0, 0.0],
    }
    result = metric.evaluate(model_output, [])
    assert result == expected


def test_reasoning_output():
    dummy_model = DummyModel()
    metric = SustainabilityMetrics(model=dummy_model)
    llm_answer = "Das ist ein Satz mit 7 Wörtern"  # 7 Wörter
    reasoning = "Ich muss ausgeben, dass der Satz 7 Wörter hat"  # 9 Wörter
    model_output = [
        LLMOutput(
            result=llm_answer, reasoning_output=reasoning, original_output=f"<think> {reasoning} </think> {llm_answer}"
        )
    ]
    expected = {
        "full_tokens": [82.0],  # (count(llm_answer) + count(reasoning) + <think> + </think>) * 4 + 10 tokens overhead
        "full_words": [18.0],  # (count(llm_answer) + count(reasoning) + <think> + </think>)
        "answer_words": [7.0],  # count(llm_answer)
        "reasoning_words": [9.0],  # count(reasoning_words)
        "energy": [0.0],
        "gwp_de": [0.0],
        "gwp_eu": [0.0],
    }
    result = metric.evaluate(model_output, [])
    assert result == expected


def test_sustainability_dense():
    mockllm = MagicMock()
    mockllm.active_parameters = 1
    mockllm.total_parameters = 1
    mockllm.tokenizer = SimpleTokenizer(mockllm)
    metric = SustainabilityMetrics(model=mockllm)
    model_output = ["Das ist sicher ein kleiner Satz mit zehn ganzen Wörtern."]
    result = metric.evaluate(model_output=model_output, labels=[])

    assert result["energy"][0] == pytest.approx(0.139, abs=1e-3)
    assert result["gwp_de"][0] == pytest.approx(0.094, abs=1e-3)
    assert result["gwp_eu"][0] == pytest.approx(0.075, abs=1e-3)


def test_sustainability_sparse():
    mockllm = MagicMock()
    mockllm.active_parameters = 10
    mockllm.total_parameters = 100
    mockllm.tokenizer = SimpleTokenizer(mockllm)
    metric = SustainabilityMetrics(model=mockllm)
    model_output = ["Das ist sicher ein kleiner Satz mit zehn ganzen Wörtern."]
    result = metric.evaluate(model_output=model_output, labels=[])

    assert result["energy"][0] == pytest.approx(0.202, abs=1e-3)
    assert result["gwp_de"][0] == pytest.approx(0.137, abs=1e-3)
    assert result["gwp_eu"][0] == pytest.approx(0.108, abs=1e-3)
