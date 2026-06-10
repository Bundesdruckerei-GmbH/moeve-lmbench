"""Contains the dataframe dataset.

The DataframeDataset is able to load the following file formats:
     - csv
     - parquet
     - json
     - jsonl
     - xml
     - xls
     - xlsx
"""

import logging
import os
from pathlib import Path
from typing import override

import pandas as pd
from pydantic import BaseModel

from lmbench.dataset.abstract import Dataset, DatasetConfig
from lmbench.models.data_models import LLMMessage, LLMRole
from lmbench.utils import get_keys_from_format_string

logger = logging.getLogger(__name__)


def dataframe_from_file(dataset_file: Path) -> pd.DataFrame:
    """Load a dataset into a pandas dataframe based on the file extension.

    Args:
        dataset_file (Path): The path to the dataset to load.

    Raises:
        ValueError: If the file format is not supported.

    Returns:
        DataFrame: The dataframe containing the dataset.
    """
    file_type = os.path.splitext(dataset_file)[1]
    match file_type:
        case ".csv":
            return pd.read_csv(dataset_file, keep_default_na=False)
        case ".parquet":
            return pd.read_parquet(dataset_file)
        case ".json":
            return pd.read_json(dataset_file)
        case ".jsonl":
            return pd.read_json(dataset_file, lines=True)
        case ".xml":
            return pd.read_xml(dataset_file)
        case ".xlsx":
            return pd.read_excel(dataset_file)
        case _:
            raise ValueError("Unsupported dataset format.")


class DataframeDatasetConfig(BaseModel):
    """The configuration for a dataframe-based dataset.

    Attributes:
        system_prompt (Optional[str]): The system prompt to use when generating prompts for the model.
        user_prompt (str): The user prompt template to use when generating prompts for the model.
        target_column (str): The column in the dataframe that contains the target values.
        shrink_column (str | list[str] | None): Column(s) that should be cut in order to keep the message smaller than
            the context size. When a list is provided, columns are shrunk sequentially in the listed order.
    """

    system_prompt: str | None = None
    user_prompt: str
    target_column: str
    shrink_column: str | list[str] | None = None
    context_column: str | None = None
    question_column: str | None = None


class DataframeDataset(Dataset):
    """A dataset that uses a pandas dataframe to represent the data."""

    dataset: pd.DataFrame
    system_prompt: str | None
    user_prompt: str
    config: DataframeDatasetConfig

    @override
    def __init__(self, name: str, dataset_config: DatasetConfig):
        super().__init__(name, dataset_config)

        # Loading the dataframe
        dataset_file = self.dataset_folder() / self.filename
        self.dataset = dataframe_from_file(dataset_file)

        self.config = DataframeDatasetConfig(**self.config_dict)

        self.system_prompt = None
        if self.config.system_prompt is not None:
            self.system_prompt = Path(self.dataset_folder() / self.config.system_prompt).read_text()
        self.user_prompt = Path(self.dataset_folder() / self.config.user_prompt).read_text()

        logger.info(f"Loaded dataset {name} with {len(self)} entries.")

    @staticmethod
    @override
    def type() -> str:
        return "dataframe"

    def _shrink_columns(self) -> list[str]:
        sc = self.config.shrink_column
        if sc is None:
            return []
        if isinstance(sc, str):
            return [sc]
        return list(sc)

    @property
    @override
    def num_shrinkable(self) -> int:
        return len(self._shrink_columns())

    def _normalize_shrink_factors(self, shrink_factor: float | list[float]) -> list[float]:
        cols = self._shrink_columns()
        if not cols:
            return []
        if isinstance(shrink_factor, list):
            padded = list(shrink_factor) + [1.0] * (len(cols) - len(shrink_factor))
            return padded[: len(cols)]
        return [shrink_factor] + [1.0] * (len(cols) - 1)

    def _apply_shrinking(self, template_vars: dict, shrink_factor: float | list[float]) -> None:
        cols = self._shrink_columns()
        if not cols:
            return
        factors = self._normalize_shrink_factors(shrink_factor)
        for col, factor in zip(cols, factors, strict=True):
            if factor >= 1.0 or col not in template_vars:
                continue
            words = str(template_vars[col]).split()
            cut = max(0, int(len(words) * factor) - 1)
            template_vars[col] = " ".join(words[:cut])

    @override
    def llm_input_at_index(self, index: int, shrink_factor: float | list[float] = 1.0) -> list[LLMMessage]:
        """
        Generate a list of LLMMessage objects for a specific row in the dataset.

        This method processes a specific row in the dataset to create a sequence of messages by formatting templates
        based on the row's content. It constructs messages for both the system prompt and the user prompt. If the system
        prompt is defined, it will be included first in the list. This method uses placeholders within the prompts that
        correspond to column names in the dataset to dynamically fill in the values from the dataset row.

        Args:
            index (int): The index of the row in the dataset for which to generate messages.
            shrink_factor (float | list[float]): The shrink factor(s) to apply (only if less than 1.0). When a list is
                provided, factors are applied per shrink column in the configured order. A scalar is applied to the
                first shrink column. Defaults to 1.0.

        Raises:
            KeyError: If any template variables are not found in the dataset row.

        Returns:
            list[LLMMessage]: A list of LLMMessage objects containing system and user prompts.
        """
        row = self.dataset.iloc[index]

        messages = []
        # Only add a system prompt if one is defined (by the `system_prompt` field in config)
        if self.system_prompt is not None:
            # Get all template variables (i.e., {varname}) in the system prompt
            system_prompt_keys = get_keys_from_format_string(self.system_prompt)
            system_message = self.system_prompt
            # If there are any template variables, fill them in from the current row of the dataset
            if len(system_prompt_keys) > 0:
                template_vars = row[system_prompt_keys].to_dict()
                self._apply_shrinking(template_vars, shrink_factor)
                system_message = system_message.format(**template_vars)  # type: ignore[reportCallIssue]
            # Add the system message to the output
            messages.append(LLMMessage(role=LLMRole.SYSTEM, content=system_message))

        # Get all template variables (i.e., {varname}) in the user prompt
        user_prompt_keys = get_keys_from_format_string(self.user_prompt)
        user_message = self.user_prompt
        # If there are any template variables, fill them in from the current row of the dataset
        if len(user_prompt_keys) > 0:
            template_vars = row[user_prompt_keys].to_dict()
            self._apply_shrinking(template_vars, shrink_factor)
            user_message = user_message.format(**template_vars)  # type: ignore[reportCallIssue]
        # Add the user message to the output
        messages.append(LLMMessage(role=LLMRole.USER, content=user_message))

        return messages

    @override
    def __len__(self) -> int:
        return len(self.dataset)

    @override
    def target_values(self, limit: int | None = None) -> list[str]:
        raw = self.dataset[self.config.target_column].to_list()
        cleaned: list[str] = []
        for i, v in enumerate(raw):
            if v is None:
                logger.error(f"Target value is None! (Index: {i})")
                cleaned.append("")
            elif not isinstance(v, str):
                logger.error(f"Wrong type of target value! (Index: {i}, Value: {v!r}, Type: {type(v)})")
                cleaned.append(str(v))
            else:
                cleaned.append(v)
        if limit is not None and limit > 0 and limit < len(cleaned):
            return cleaned[:limit]
        return cleaned

    def get_column(self, name: str) -> list:
        """Return the specified dataset column as a list."""
        return self.dataset[name].tolist()
