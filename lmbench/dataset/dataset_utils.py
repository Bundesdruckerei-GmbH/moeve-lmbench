"""This module contains functions to load a dataset and create the necessary dataset class based on a dataset name.

When the `load_dataset` function is called, the presence of the dataset in the `DATA_FOLDER` is checked. If the dataset
is not present, it will be downloaded from the appropriate container in the azure blob storage as configured in the
config file. This mechanism is only used for local development and testing purposes. In production environment, datasets
container are expected to be mounted in the `DATA_FOLDER` directory.

Based on the config.yaml in the corresponding dataset folder, the type of task (e.g., summarization, question answering,
etc.) and other relevant information such as input/output modalities, etc. will be determined. Also the appropriate
dataset class for that particular task is instantiated based on the "type"-field in the configuration.
"""

import logging
import os
import shutil

from omegaconf import OmegaConf

from lmbench.config.config import DATASET_FOLDER
from lmbench.dataset.abstract import Dataset, DatasetConfig
from lmbench.dataset.dataframe import DataframeDataset

DATASET_LIST: list[type[Dataset]] = [DataframeDataset]
DATASET_MAP = {d.type(): d for d in DATASET_LIST}

logger = logging.getLogger(__name__)


def download_dataset(name: str) -> None:
    """Download the dataset, using Azure Blob Storage if available.

    Args:
        name: The name of the dataset to be downloaded.

    Raises:
        FileNotFoundError: If the dataset is not found locally and Azure support is not available.
    """
    try:
        from lmbench.azure_storage import (  # pyright: ignore[reportMissingImports]
            download_dataset as azure_download_dataset,
        )

        azure_download_dataset(name)
    except ImportError:
        raise FileNotFoundError(
            f"Dataset '{name}' not found in '{DATASET_FOLDER}'. "
            f"Please place the dataset files in '{DATASET_FOLDER}/{name}/'."
        ) from None


def load_dataset_config(name: str) -> DatasetConfig:
    """Load the config for a given dataset from the dataset folder.

    This function assumes that the dataset is already present in the `DATA_FOLDER` and does not handle downloading of
    datasets.

    Args:
        name (str): The name of the dataset to be loaded.

    Returns:
        DatasetConfig: The config for the given dataset.
    """
    loaded_conf = OmegaConf.load(DATASET_FOLDER / name / "config.yaml")
    conf_dict = OmegaConf.to_object(loaded_conf)
    config: DatasetConfig = DatasetConfig(**conf_dict)  # type: ignore
    return config


def load_dataset(name: str, force_download: bool) -> Dataset:
    """Loads a dataset from the configured data folder or downloads it if not present.

    Args:
        name (str): The name of the dataset to be loaded.
        force_download (bool): If it exists, removes downloaded dataset to enable new download.
    """
    dataset_path = DATASET_FOLDER / name

    if force_download:
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)

    if not os.path.exists(dataset_path):
        logger.info(f"Dataset {name} not found. Now downloading...")
        download_dataset(name)

    dataset_config = load_dataset_config(name)
    dataset_class = DATASET_MAP[dataset_config.dataset_type]
    dataset = dataset_class(name=name, dataset_config=dataset_config)
    return dataset
