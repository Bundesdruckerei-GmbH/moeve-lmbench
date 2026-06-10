"""This module contains the abstract class for benchmark_datasets."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from lmbench.config.config import DATASET_FOLDER
from lmbench.models.data_models import LLMMessage
from lmbench.task import Task


class DatasetConfig(BaseModel):
    """General dataset configuration that is used for every dataset.

    Individual configurations for specific dataset implemenations can use the dataset_config dictionary.

    Attributes:
        task (Task): The task this dataset is meant to be used for.
        filename (str): The filename of the dataset inside the dataset folder
        dataset_type (str): The type of the dataset (e.g., 'dataframe')
        config (dict[str, Any]): Additional configuration for the specific dataset implementation.
    """

    task: Task
    filename: str
    dataset_type: str
    config: dict[str, Any] = {}


class Dataset(ABC):
    """Abstract class for datasets."""

    """The name of the dataset in human-readable format without spaces."""
    name: str
    """The task that the dataset is used for. """
    task: Task

    def __init__(self, name: str, dataset_config: DatasetConfig):
        """Initialize the dataset with the given name, task and reference_column.

        Args:
            name (str): The name of the dataset in human-readable format without spaces.
            dataset_config (DatasetConfig): The configuration object for this dataset.
        """
        self.name = name

        self.task = dataset_config.task
        self.filename = dataset_config.filename
        self.config_dict = dataset_config.config

    @staticmethod
    def type() -> str:
        """Return the type of the dataset as a string. With this type the dataset class is identified that loads the
        data.

        Returns:
            str: The type of the dataset as a string.
        """
        raise NotImplementedError

    def llm_input_at_index(self, index: int, shrink_factor: float | list[float] = 1.0) -> list[LLMMessage]:
        """Returns the input for the LLM at the given index in the format of a list of `LLMMessage`.

        Args:
            index (int): Index of the row to get the input for.
            shrink_factor (float | list[float]): Factor(s) by which to shrink the input. Only values smaller than 1.0
              cause shrinking. When the dataset supports multiple shrinkable columns, a list applies one factor per
              column in the configured order. A scalar is applied to the first shrinkable column. Defaults to 1.0.

        Returns:
            list[LLMMessage]: The input for the LLM as a list of `LLMMessage` objects.
        """
        raise NotImplementedError

    @property
    def num_shrinkable(self) -> int:
        """Number of independently shrinkable input dimensions. Defaults to 0 (no shrinking supported)."""
        return 0

    def target_values(self, limit: int | None = None) -> list[str]:
        """Return the list of target values of the dataset with the given limit.

        These references can be compared with the actual llm outputs in order to calculate metrics.
        If the dataset has no target column, returns a list of empty strings of the appropriate length.

        Args:
            limit (Optional[int]): To how many entries the target values should be limited. If None, all values are
                returned. Defaults to None.

        Returns:
            list[str]: A list of "gold standard" target values of the dataset, beginning with the first entry.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Returns the length of the dataset, i.e., the number of rows in the dataset.

        Returns:
            int: Number of rows in the dataset
        """
        raise NotImplementedError

    @abstractmethod
    def get_column(self, name: str) -> Sequence:
        """
        Return the sequence of values for a named column or metadata field.
        If unsupported, raise NotImplementedError.
        """
        raise NotImplementedError

    def dataset_folder(self) -> Path:
        """Returns the path to the folder where the dataset is stored.

        Returns:
            Path: Path to the folder where the dataset is stored.
        """
        return Path(os.path.join(DATASET_FOLDER, f"{self.name}"))
