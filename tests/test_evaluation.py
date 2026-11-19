import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

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
    dataset.generation_indices = MagicMock(return_value=[None] * 10)

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


def _stamped_dataset(tmp_path, mocker):
    """A real DataframeDataset over a small stamped dataset file on disk."""
    from lmbench.dataset.abstract import ITEM_ID_COLUMN, DatasetConfig
    from lmbench.dataset.dataframe import DataframeDataset, DataframeDatasetConfig
    from lmbench.task import Task

    folder = tmp_path / "e2e"
    folder.mkdir()
    pd.DataFrame(
        {
            "shrink_column": ["doc a", "doc b", "doc c"],
            "target_column": ["target0", "target1", "target2"],
            ITEM_ID_COLUMN: ["e2e:000", "e2e:001", "e2e:002"],
        }
    ).to_csv(folder / "e2e.csv", index=False)
    (folder / "user_prompt.txt").write_text("a prompt")
    mocker.patch("lmbench.dataset.abstract.DATASET_FOLDER", str(tmp_path))
    config = DatasetConfig(
        task=Task.SUMMARIZATION,
        filename="e2e.csv",
        dataset_type="dataframe",
        config=DataframeDatasetConfig(
            system_prompt=None,
            user_prompt="user_prompt.txt",
            target_column="target_column",
            shrink_column="shrink_column",
        ).model_dump(),
    )
    return DataframeDataset("e2e", dataset_config=config)


def test_run_writes_a_per_item_artifact_and_stamps_output_csv(tmp_path, mocker):
    import json

    import pyarrow.parquet as pq

    from lmbench.per_item import ARTIFACT_FILENAME

    dataset = _stamped_dataset(tmp_path, mocker)

    llm = MagicMock()
    llm.model = "test-model"
    llm.temperature = 0.0
    llm.check_tokens = MagicMock(return_value=1.0)
    # Two hits and one miss, so the artifact has to carry both an ok and a scored-zero row.
    llm.cached_invoke = MagicMock(side_effect=["target0", "target1", "something else"])

    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("lmbench.evaluation.OUTPUT_FOLDER", temp_dir):
            runner = EvaluationRunner(llm=llm, dataset=dataset)
            runner.run([Match], use_metrics_cache=False)

            output = pd.read_csv(os.path.join(temp_dir, "output.csv"))
            assert output.columns.to_list()[0] == "item_id"
            assert output["item_id"].to_list() == ["e2e:000", "e2e:001", "e2e:002"]

            table = pq.read_table(os.path.join(temp_dir, ARTIFACT_FILENAME))

    rows = table.to_pylist()
    assert len(rows) == 3 * 3  # three items x three ExactMatch sub-scores
    assert {row["metric"] for row in rows} == {
        "ExactMatch:exact_match",
        "ExactMatch:case_insensitive_match",
        "ExactMatch:fuzzy_match",
    }
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["judge"] is None for row in rows)
    # A plain summarization dataset has no resampling structure above the item.
    assert all(row["cluster_id"] is None and row["cluster_id_2"] is None and row["stratum"] is None for row in rows)
    assert all(row["generation_index"] is None for row in rows)

    exact = {row["item_id"]: row["score"] for row in rows if row["metric"] == "ExactMatch:exact_match"}
    assert exact == {"e2e:000": 1.0, "e2e:001": 1.0, "e2e:002": 0.0}

    metadata = {key.decode(): value.decode() for key, value in table.schema.metadata.items()}
    assert metadata["schema_version"] == "1"
    assert metadata["dataset"] == "e2e"
    assert metadata["task"] == "summarization"
    assert metadata["n_items"] == "3"
    assert metadata["dataset_version"].startswith("sha256:")
    assert metadata["n_rows_evaluated"] == "3"
    assert json.loads(metadata["selected_indices"]) == [0, 1, 2]
    assert json.loads(metadata["covered_metrics"]) == sorted(
        ["ExactMatch:exact_match", "ExactMatch:case_insensitive_match", "ExactMatch:fuzzy_match"]
    )
    assert json.loads(metadata["metrics_without_per_item"]) == []


def test_skip_generation_without_selected_indices_uses_cached_order(tmp_path, mocker):
    """Runs archived before per-item scoring have generations.parquet but no selected_indices.json;
    skip_generation falls back to the cached order in the file instead of erroring."""
    dataset = _stamped_dataset(tmp_path, mocker)

    llm = MagicMock()
    llm.model = "test-model"
    llm.temperature = 0.0

    with tempfile.TemporaryDirectory() as temp_dir:
        pd.DataFrame(["target0", "target1", "wrong"]).to_parquet(os.path.join(temp_dir, "generations.parquet"))
        with patch("lmbench.evaluation.OUTPUT_FOLDER", temp_dir):
            EvaluationRunner(llm=llm, dataset=dataset).run([Match], skip_generation=True, use_metrics_cache=False)

            assert not os.path.exists(os.path.join(temp_dir, "selected_indices.json"))
            output = pd.read_csv(os.path.join(temp_dir, "output.csv"))
            assert output["output"].to_list() == ["target0", "target1", "wrong"]


def test_skip_generation_count_mismatch_still_raises(tmp_path, mocker):
    """The count guard survives the missing-file fallback: fewer cached rows than selected still fails."""
    dataset = _stamped_dataset(tmp_path, mocker)

    llm = MagicMock()
    llm.model = "test-model"

    with tempfile.TemporaryDirectory() as temp_dir:
        pd.DataFrame(["only", "two"]).to_parquet(os.path.join(temp_dir, "generations.parquet"))
        with patch("lmbench.evaluation.OUTPUT_FOLDER", temp_dir):
            with pytest.raises(AssertionError):
                EvaluationRunner(llm=llm, dataset=dataset).run([Match], skip_generation=True, use_metrics_cache=False)


def test_run_without_item_ids_fails(tmp_path, mocker):
    from lmbench.dataset.abstract import ITEM_ID_COLUMN, MissingItemIdError

    dataset = _stamped_dataset(tmp_path, mocker)
    dataset.dataset = dataset.dataset.drop(columns=[ITEM_ID_COLUMN])

    llm = MagicMock()
    llm.model = "test-model"
    llm.check_tokens = MagicMock(return_value=1.0)
    llm.cached_invoke = MagicMock(return_value="whatever")

    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("lmbench.evaluation.OUTPUT_FOLDER", temp_dir):
            with pytest.raises(MissingItemIdError):
                EvaluationRunner(llm=llm, dataset=dataset).run([Match], use_metrics_cache=False)


def test_content_filter_rejection_is_recorded_instead_of_failing_the_run():
    from lmbench.models.exceptions import CONTENT_FILTERED_OUTPUT, ContentFilterException

    runner, _, llm = _make_runner(num_shrinkable=1, token_factors=[1.0, 1.0])
    llm.cached_invoke = MagicMock(side_effect=[ContentFilterException("filtered"), "ok"])
    assert runner._compute_generations([0, 1], num_threads=1, use_cache=True) == [CONTENT_FILTERED_OUTPUT, "ok"]


def test_cache_key_suffix_follows_a_generation_index_column():
    """A dataframe dataset that repeats single rows gets one cache entry per repeat; 0 and empty keep the plain key."""
    runner, dataset, llm = _make_runner(num_shrinkable=1, token_factors=[1.0] * 4)
    dataset.generation_indices = MagicMock(return_value=[None, 0, 1, 4])
    llm.cached_invoke = MagicMock(return_value="ok")
    runner._compute_generations([0, 1, 2, 3], num_threads=1, use_cache=True)
    suffixes = [c.kwargs["cache_key_suffix"] for c in llm.cached_invoke.call_args_list]
    assert suffixes == [None, None, "g=1", "g=4"]
