import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd

from lmbench.evaluation import EvaluationRunner
from lmbench.metrics.match import Match
from lmbench.models.exceptions import ContextSizeException


def test_evaluation():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use patch to mock OUTPUT_FOLDER in the lmbench.evaluation module
        with patch("lmbench.evaluation.OUTPUT_FOLDER", temp_dir):
            evaluation = EvaluationRunner(llm=MagicMock(), dataset=MagicMock())
            evaluation.run([Match], use_metrics_cache=False)

            output_file_path = os.path.join(temp_dir, "output.csv")
            result = pd.read_csv(output_file_path)

            assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
            assert "target" in result.columns.to_list(), "DataFrame should have target column"
            assert "output" in result.columns.to_list(), "DataFrame should have output column"
            assert len(result) == 0, "DataFrame should be empty initially"


def _make_runner(num_shrinkable: int, token_factors: list[float]):
    """Build a runner whose dataset reports `num_shrinkable` shrink columns and whose
    llm.check_tokens returns the next value from `token_factors` on each call.
    """
    dataset = MagicMock()
    dataset.num_shrinkable = num_shrinkable
    dataset.llm_input_at_index = MagicMock(return_value=[{"role": "user", "content": "x"}])

    llm = MagicMock()
    llm.check_tokens = MagicMock(side_effect=token_factors)
    llm.invoke = MagicMock(return_value="ok")

    runner = EvaluationRunner(llm=llm, dataset=dataset)
    return runner, dataset, llm


def _shrink_factor_calls(dataset_mock: MagicMock) -> list:
    """Extract the shrink_factor argument from each llm_input_at_index call that passed one."""
    calls = []
    for call in dataset_mock.llm_input_at_index.call_args_list:
        if "shrink_factor" in call.kwargs:
            calls.append(call.kwargs["shrink_factor"])
    return calls


def test_compute_generations_no_shrink_needed():
    """If the initial prompt fits, no shrinking calls are made."""
    runner, dataset, _ = _make_runner(num_shrinkable=3, token_factors=[1.0])
    result = runner._compute_generations([0], num_threads=1, use_cache=False)
    assert result == ["ok"]
    assert _shrink_factor_calls(dataset) == []


def test_compute_generations_single_column_backcompat():
    """num_shrinkable=1 reduces the single factor multiplicatively (matches legacy behavior)."""
    # initial check returns 2.0 (over by 2x), after shrink to [0.5] returns 1.0 (fits).
    runner, dataset, _ = _make_runner(num_shrinkable=1, token_factors=[2.0, 1.0])
    runner._compute_generations([0], num_threads=1, use_cache=False)
    assert _shrink_factor_calls(dataset) == [[0.5]]


def test_compute_generations_first_column_sufficient():
    """When shrinking only the first column suffices, later columns stay at 1.0 (cache-compat)."""
    runner, dataset, _ = _make_runner(num_shrinkable=3, token_factors=[2.0, 1.0])
    runner._compute_generations([0], num_threads=1, use_cache=False)
    assert _shrink_factor_calls(dataset) == [[0.5, 1.0, 1.0]]


def test_compute_generations_advances_to_next_column():
    """When the active column hits epsilon, the loop advances to the next column."""
    # Each shrink barely helps (token_factor stays high) until factor underflows EPSILON and we advance.
    # token_factor sequence: initial=2.0, then keep returning 2.0 until first column is exhausted,
    # then 1.5 (advanced to col 1), then 1.0 (fits).
    token_factors = [2.0] * 25 + [1.5, 1.0]
    runner, dataset, _ = _make_runner(num_shrinkable=3, token_factors=token_factors)
    runner._compute_generations([0], num_threads=1, use_cache=False)

    factor_calls = _shrink_factor_calls(dataset)
    # The final call's factors should have col 0 fully consumed and col 1 partially shrunk.
    last = factor_calls[-1]
    assert last[0] == 0.0, "first column should be exhausted before advancing"
    assert 0.0 < last[1] < 1.0, "second column should be partially shrunk"
    assert last[2] == 1.0, "third column should be untouched"


def test_compute_generations_context_size_exception_retry():
    """ContextSizeException triggers a last-chance shrink on the active column and retries."""
    dataset = MagicMock()
    dataset.num_shrinkable = 3
    dataset.llm_input_at_index = MagicMock(return_value=[{"role": "user", "content": "x"}])

    llm = MagicMock()
    llm.check_tokens = MagicMock(return_value=1.0)  # initial fit; exception comes from invoke
    llm.invoke = MagicMock(side_effect=[ContextSizeException(shrink_factor=0.5, message="too big"), "ok"])

    runner = EvaluationRunner(llm=llm, dataset=dataset)
    result = runner._compute_generations([0], num_threads=1, use_cache=False)

    assert result == ["ok"]
    factor_calls = _shrink_factor_calls(dataset)
    # Exactly one retry call with shrink_factor applied to the active (first) column.
    assert len(factor_calls) == 1
    retry_factors = factor_calls[0]
    assert retry_factors[0] < 0.5  # 0.5 * 1.0 minus the 0.01 last-chance shrinkage
    assert retry_factors[1] == 1.0
    assert retry_factors[2] == 1.0
