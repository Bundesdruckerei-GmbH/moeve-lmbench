import pandas as pd
import pytest
from pytest_mock import MockFixture

from lmbench.dataset.abstract import ITEM_ID_COLUMN, DatasetConfig, MissingItemIdError
from lmbench.dataset.dataframe import DataframeDataset, DataframeDatasetConfig
from lmbench.dataset.multi_iteration import MultiIterationDataset
from lmbench.models.data_models import LLMMessage, LLMRole
from lmbench.task import Task

dataframe_config = DataframeDatasetConfig(
    system_prompt=None,
    user_prompt="user_prompt.txt",
    target_column="target_column",
    shrink_column="shrink_column",
)

dataset_config = DatasetConfig(
    task=Task.SUMMARIZATION, filename="testfile.csv", dataset_type="dataframe", config=dataframe_config.model_dump()
)


def build_dataset(mocker: MockFixture, df: pd.DataFrame) -> DataframeDataset:
    mocker.patch("lmbench.dataset.dataframe.dataframe_from_file", return_value=df)
    mocker.patch("pathlib.Path.read_text", return_value="a prompt")
    return DataframeDataset("test_dataset", dataset_config=dataset_config)


def stamped_frame(rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shrink_column": [" ".join(["word"] * 20)] * rows,
            "target_column": [f"target{i}" for i in range(rows)],
            ITEM_ID_COLUMN: [f"test_dataset:{i:03d}" for i in range(rows)],
        }
    )


def test_item_ids(mocker: MockFixture):
    dataset = build_dataset(mocker, stamped_frame())
    assert dataset.item_ids() == ["test_dataset:000", "test_dataset:001", "test_dataset:002"]


def test_item_ids_with_limit(mocker: MockFixture):
    dataset = build_dataset(mocker, stamped_frame())
    assert dataset.item_ids(limit=2) == ["test_dataset:000", "test_dataset:001"]
    # A limit at or above the row count returns everything, matching target_values.
    assert dataset.item_ids(limit=3) == dataset.item_ids()
    assert dataset.item_ids(limit=10) == dataset.item_ids()


def test_item_ids_are_unique(mocker: MockFixture):
    dataset = build_dataset(mocker, stamped_frame(rows=50))
    ids = dataset.item_ids()
    assert len(set(ids)) == len(ids)


def test_missing_item_id_column_raises(mocker: MockFixture):
    dataset = build_dataset(mocker, stamped_frame().drop(columns=[ITEM_ID_COLUMN]))
    with pytest.raises(MissingItemIdError):
        dataset.item_ids()


def test_item_ids_survive_shrink_factors(mocker: MockFixture):
    """Shrinking rewrites prompt text only; ids are a property of the row and must not move."""
    dataset = build_dataset(mocker, stamped_frame())
    dataset.user_prompt = "{shrink_column}"
    expected = dataset.item_ids()

    for factor in [1.0, 0.5, 0.1, 0.0]:
        for index in range(len(dataset)):
            dataset.llm_input_at_index(index, factor)
        assert dataset.item_ids() == expected


def test_item_id_is_not_rendered_into_prompts(mocker: MockFixture):
    """The prompt only reads its own template keys, so a stamped column cannot leak into the LLM input."""
    dataset = build_dataset(mocker, stamped_frame())
    dataset.user_prompt = "{target_column}"
    assert dataset.llm_input_at_index(0) == [LLMMessage(role=LLMRole.USER, content="target0")]


def test_item_ids_select_by_indices(mocker: MockFixture):
    """Callers subset ids the way EvaluationRunner subsets target values."""
    dataset = build_dataset(mocker, stamped_frame(rows=5))
    selected_indices = [0, 2, 4]
    ids = dataset.item_ids(limit=None)
    assert [ids[i] for i in selected_indices] == ["test_dataset:000", "test_dataset:002", "test_dataset:004"]


def test_multi_iteration_repeats_item_ids(mocker: MockFixture):
    """Repeats are interleaved, matching MultiIterationDataset's `index % len(base)` arithmetic."""
    base = build_dataset(mocker, stamped_frame())
    dataset = MultiIterationDataset(base, 2)

    ids = dataset.item_ids()
    assert len(ids) == len(dataset)
    assert ids == ["test_dataset:000", "test_dataset:001", "test_dataset:002"] * 2

    # Each id lines up with the base row that produced the prompt at that index.
    for index in range(len(dataset)):
        assert ids[index] == base.item_ids()[index % len(base)]


def test_multi_iteration_item_ids_with_limit(mocker: MockFixture):
    base = build_dataset(mocker, stamped_frame())
    dataset = MultiIterationDataset(base, 2)
    assert dataset.item_ids(limit=4) == ["test_dataset:000", "test_dataset:001", "test_dataset:002", "test_dataset:000"]
    assert dataset.item_ids(limit=10) == dataset.item_ids()


def test_multi_iteration_propagates_missing_column(mocker: MockFixture):
    base = build_dataset(mocker, stamped_frame().drop(columns=[ITEM_ID_COLUMN]))
    dataset = MultiIterationDataset(base, 2)
    with pytest.raises(MissingItemIdError):
        dataset.item_ids()


def test_generation_indices_without_column_are_none(mocker: MockFixture):
    dataset = build_dataset(mocker, stamped_frame())
    assert dataset.generation_indices() == [None, None, None]


def test_generation_indices_come_from_the_optional_column(mocker: MockFixture):
    frame = stamped_frame(rows=4)
    frame["generation_index"] = pd.array([None, 0, 1, 2], dtype="Int64")
    dataset = build_dataset(mocker, frame)
    assert dataset.generation_indices() == [None, 0, 1, 2]
    assert dataset.generation_indices(limit=2) == [None, 0]
