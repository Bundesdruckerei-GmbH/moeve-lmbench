import pytest

from lmbench.metrics.output_status import model_output_is_broken


@pytest.mark.parametrize(
    "output",
    [
        "",
        "   \n\t ",
        "<think>the model reasoned but never answered</think>",
        "<think>reasoning that never closed",
        "<|channel|>final<|message|>leaked harmony markers",
        "assistantfinal never unwrapped",
    ],
)
def test_broken_outputs(output):
    assert model_output_is_broken(output) is True


@pytest.mark.parametrize(
    "output",
    [
        "Berlin is the capital of Germany.",
        "<think>reasoning</think>Berlin is the capital.",
        "42",
    ],
)
def test_usable_outputs(output):
    assert model_output_is_broken(output) is False
