import os

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockFixture

from lmbench.dataset.abstract import DatasetConfig
from lmbench.dataset.dataframe import DataframeDataset, DataframeDatasetConfig, dataframe_from_file
from lmbench.models.data_models import LLMMessage, LLMRole
from lmbench.task import Task

dataframe_config = DataframeDatasetConfig(
    system_prompt="system_prompt.txt",
    user_prompt="user_prompt.txt",
    target_column="target_column",
    shrink_column="shrink_column",
)

dataset_config = DatasetConfig(
    task=Task.SUMMARIZATION, filename="testfile.csv", dataset_type="dataframe", config=dataframe_config.model_dump()
)


def test_dataframe_from_file():
    valid_file_extensions = [".csv", ".parquet", ".json", ".xml", ".xlsx"]
    df = pd.DataFrame()
    df["a"] = [1, 2, 3]
    df["b"] = [5, 6, 7]
    for ext in valid_file_extensions:
        file_path = f"test_dataframe_from_file{ext}"
        match ext:
            case ".csv":
                df.to_csv(file_path, index=False)
            case ".parquet":
                df.to_parquet(file_path, index=False)
            case ".json":
                df.to_json(file_path, index=False)
            case ".xml":
                df.to_xml(file_path, index=False)
            case ".xlsx":
                df.to_excel(file_path, index=False)
        loaded_df = dataframe_from_file(file_path)
        assert len(loaded_df) == 3, "Loaded DataFrame should have 3 rows"
        assert loaded_df.columns.tolist() == ["a", "b"], "DataFrame should have columns 'a' and 'b'"
        os.remove(file_path)
    with pytest.raises(ValueError):
        dataframe_from_file("test_dataframe_from_file.txt")


def test_simple_dataset(mocker: MockFixture):
    """Test loading a dataset."""
    df = pd.DataFrame()
    df["a"] = [1, 2, 3]
    df["b"] = [5, 6, 7]
    mocker.patch("lmbench.dataset.dataframe.dataframe_from_file", return_value=df)
    mocker.patch("pathlib.Path.read_text", return_value="a prompt")
    dataset = DataframeDataset("test_dataset", dataset_config=dataset_config)
    assert dataset.task == Task.SUMMARIZATION, "Dataset task should be summarization"
    assert dataset.type() == "dataframe", "Dataset type should be 'dataframe'"
    assert dataset.system_prompt == "a prompt", "System prompt should be 'a prompt'"
    assert dataset.user_prompt == "a prompt", "User prompt should be 'a prompt'"
    assert dataset.filename == "testfile.csv", "Filename should be 'testfile.csv'"
    assert len(dataset) == 3, "Dataset length should be 3"
    assert dataset.dataset.to_dict() == df.to_dict(), "Dataset should match the original DataFrame"
    llm_messages = [
        LLMMessage(role=LLMRole.SYSTEM, content="a prompt"),
        LLMMessage(role=LLMRole.USER, content="a prompt"),
    ]
    assert dataset.llm_input_at_index(0) == llm_messages, "LLM input at index 0 should match expected messages"


def test_filling_prompt(mocker: MockFixture):
    df = pd.DataFrame()
    df["a"] = [1, 2, 3]
    df["b"] = [5, 6, 7]
    mocker.patch("lmbench.dataset.dataframe.dataframe_from_file", return_value=df)
    mocker.patch("pathlib.Path.read_text", return_value="a prompt")
    dataset = DataframeDataset("test_dataset", dataset_config=dataset_config)

    dataset.system_prompt = None
    dataset.user_prompt = "Insert a: {a}, insert b: {b}"
    for i in range(3):
        llm_messages = [
            LLMMessage(role=LLMRole.USER, content=f"Insert a: {df.loc[i]['a']}, insert b: {df.loc[i]['b']}")
        ]
        assert dataset.llm_input_at_index(i) == llm_messages, "LLM input at index should match expected messages"


def test_target_column(mocker: MockFixture):
    df = pd.DataFrame()
    df["a"] = [1, 2, 3]
    df["target_column"] = ["a", "b", "c"]
    mocker.patch("lmbench.dataset.dataframe.dataframe_from_file", return_value=df)
    mocker.patch("pathlib.Path.read_text", return_value="a prompt")
    dataset = DataframeDataset("test_dataset", dataset_config=dataset_config)

    assert dataset.target_values() == ["a", "b", "c"], "Target values should be ['a', 'b', 'c']"
    assert dataset.target_values(1) == ["a"], "Target value at index 1 should be ['a']"


def test_shrinking(mocker: MockFixture):
    df = pd.DataFrame()
    long_text = "Das ist ein Test " * 10
    df["shrink_column"] = [long_text, "Das ist ein Test", "Hallo"]
    df["b"] = [5, 6, 7]
    mocker.patch("lmbench.dataset.dataframe.dataframe_from_file", return_value=df)
    mocker.patch("pathlib.Path.read_text", return_value="a prompt")
    dataset = DataframeDataset("test_dataset", dataset_config=dataset_config)

    def shrink_text(text, factor):
        return " ".join(text.split()[: int(len(text.split()) * factor) - 1])

    dataset.system_prompt = None
    dataset.user_prompt = "{shrink_column}"
    for shrink in np.arange(0.1, 1.0, 0.1):
        assert dataset.llm_input_at_index(0, float(shrink)) == [
            LLMMessage(role=LLMRole.USER, content=shrink_text(long_text, shrink))
        ], "LLM input with shrink should match expected messages"


def _shrink_text(text: str, factor: float) -> str:
    words = text.split()
    cut = max(0, int(len(words) * factor) - 1)
    return " ".join(words[:cut])


def test_shrinking_to_empty(mocker: MockFixture):
    """A factor small enough must reduce a column to the empty string (no all-but-last bug)."""
    df = pd.DataFrame()
    long_text = " ".join(["word"] * 100)
    df["shrink_column"] = [long_text]
    mocker.patch("lmbench.dataset.dataframe.dataframe_from_file", return_value=df)
    mocker.patch("pathlib.Path.read_text", return_value="a prompt")
    dataset = DataframeDataset("test_dataset", dataset_config=dataset_config)

    dataset.system_prompt = None
    dataset.user_prompt = "{shrink_column}"
    assert dataset.llm_input_at_index(0, 0.0) == [LLMMessage(role=LLMRole.USER, content="")]
    assert dataset.llm_input_at_index(0, 0.005) == [LLMMessage(role=LLMRole.USER, content="")]


def test_shrinking_multi_column_list(mocker: MockFixture):
    """List-shaped shrink_factor shrinks each configured column with its own factor."""
    multi_config = DataframeDatasetConfig(
        system_prompt=None,
        user_prompt="user_prompt.txt",
        target_column="target_column",
        shrink_column=["col_a", "col_b", "col_c"],
    )
    multi_dataset_config = DatasetConfig(
        task=Task.SUMMARIZATION,
        filename="testfile.csv",
        dataset_type="dataframe",
        config=multi_config.model_dump(),
    )

    text_a = " ".join(["a"] * 50)
    text_b = " ".join(["b"] * 50)
    text_c = " ".join(["c"] * 50)
    df = pd.DataFrame({"col_a": [text_a], "col_b": [text_b], "col_c": [text_c]})
    mocker.patch("lmbench.dataset.dataframe.dataframe_from_file", return_value=df)
    mocker.patch("pathlib.Path.read_text", return_value="a prompt")
    dataset = DataframeDataset("test_dataset", dataset_config=multi_dataset_config)

    assert dataset.num_shrinkable == 3

    dataset.system_prompt = None
    dataset.user_prompt = "{col_a}|{col_b}|{col_c}"

    # Only first column shrunk — preserves backwards compat for runs that only needed doc 1 shrinking.
    msgs = dataset.llm_input_at_index(0, [0.5, 1.0, 1.0])
    assert msgs == [LLMMessage(role=LLMRole.USER, content=f"{_shrink_text(text_a, 0.5)}|{text_b}|{text_c}")]

    # First exhausted, second partially shrunk.
    msgs = dataset.llm_input_at_index(0, [0.0, 0.5, 1.0])
    assert msgs == [LLMMessage(role=LLMRole.USER, content=f"|{_shrink_text(text_b, 0.5)}|{text_c}")]

    # All three at zero.
    msgs = dataset.llm_input_at_index(0, [0.0, 0.0, 0.0])
    assert msgs == [LLMMessage(role=LLMRole.USER, content="||")]

    # Scalar factor applies to first column only (backwards compat with old API).
    msgs = dataset.llm_input_at_index(0, 0.5)
    assert msgs == [LLMMessage(role=LLMRole.USER, content=f"{_shrink_text(text_a, 0.5)}|{text_b}|{text_c}")]

    # Short list is padded with 1.0 for unspecified columns.
    msgs = dataset.llm_input_at_index(0, [0.5])
    assert msgs == [LLMMessage(role=LLMRole.USER, content=f"{_shrink_text(text_a, 0.5)}|{text_b}|{text_c}")]
