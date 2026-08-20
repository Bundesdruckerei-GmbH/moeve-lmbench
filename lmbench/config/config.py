"""
This module contains the configuration for the logging system and references to the project root
directory used throughout the project.
"""

import os
from pathlib import Path
from typing import Any

import dotenv
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

dotenv.load_dotenv()


class LMArgs(BaseModel):
    """Generation parameters for an LM client.

    Lives as a dedicated sub-block on ``JudgeLLMConfig`` to keep
    connection/auth fields and generation knobs visibly separated. Fields
    cover the standard OpenAI Chat Completions params; backend-specific
    extensions (e.g. vLLM's ``min_p``, ``top_k``, ``chat_template_kwargs``)
    go under ``extra_body``, which the OpenAI SDK merges verbatim into the
    request body. Consumers typically dump non-``None`` values via
    ``model_dump(exclude_none=True)`` and spread them into the client.
    """

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    seed: int | None = None
    stop: list[str] | None = None
    extra_body: dict[str, Any] | None = None


class JudgeLLMConfig(BaseModel):
    """Configuration for an LLM used as a judge by metrics (ragas, values, hallucination, etc.).

    Connection/auth info lives at the top level; generation parameters live
    under ``lm_args`` so the two concerns are visibly separated. ``lm_args``
    is consumed by metrics that build a model-side client directly (currently
    the hallucination metric via dspy); ragas/values metrics ignore it.

    Attributes:
        provider: The provider type: "openai" or "azure_openai".
        model: The model name (e.g., "gpt-4o-mini").
        base_url: Base URL for the API (required for openai provider, including OpenAI-compatible endpoints).
        api_key: Resolved API key (OmegaConf substitutes the env var before this is read).
        azure_endpoint: Azure endpoint URL (only for azure_openai provider).
        api_version: API version (Azure OpenAI, or as a query param for Azure AI Foundry).
        azure_deployment: Azure deployment name (only for azure_openai provider).
        lm_args: Generation parameters (temperature, max_tokens, sampling extensions, etc.).
    """

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    base_url: str = ""
    api_key: str = ""
    azure_endpoint: str = ""
    api_version: str = ""
    azure_deployment: str = ""
    lm_args: LMArgs = Field(default_factory=LMArgs)


class SemScoreConfig(BaseModel):
    """Config for the SemScore metric.

    Attributes:
        embedding_model_name (str): The name of the model used for generating embeddings.
        embedding_model_ctx_size (int): The context size of the embedding model.
    """

    embedding_model_name: str
    embedding_model_ctx_size: int
    chunk_size: int


class RetryConfig(BaseModel):
    """Configuration of Retry.

    Attributes:
        stop_after_attempt (int): The number of attempts after which the retry will stop.
        min_wait (int): The minimum wait time between retries in seconds.
        max_wait (int): The maximum wait time between retries in seconds.
    """

    stop_after_attempt: int
    min_wait: int
    max_wait: int


class HallucinationConfig(BaseModel):
    """Hallucination-metric-specific hyperparameters (not LLM-connection settings).

    LLM connection/sampling settings live in ``JudgeLLMConfig`` (see
    ``Config.hallucination_judge_llm``). This block carries only what is
    meaningful for the metric's scoring logic.

    Attributes:
        judge_n: Number of judge samples per evaluated sub-metric (multi-shot averaging).
            ``judge_n > 1`` additionally emits per-sub-metric variance diagnostics.
        answerable_weights: Sub-metric weights used for ANSWERABLE rows.
        unanswerable_weights: Sub-metric weights used for UNANSWERABLE rows.
        conflicting_weights: Sub-metric weights used for CONFLICTING rows.
    """

    judge_n: int = 3
    answerable_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "status_correctness": 0.20,
            "factual_correctness": 0.30,
            "document_f1": 0.30,
            "context_groundedness": 0.20,
        }
    )
    unanswerable_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "status_correctness": 0.30,
            "refusal_quality": 0.40,
            "knowledge_leakage": 0.30,
        }
    )
    conflicting_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "status_correctness": 0.30,
            "conflict_detection_quality": 0.40,
            "document_f1": 0.30,
        }
    )


class Config(BaseModel):
    """Config of LMBench.

    Attributes:
        dataset_folder (str): The folder where the datasets are stored. Relative to the root folder of the project.
        tokenizer_folder (str): The folder where the tokenizers are stored. Relative to the root folder of the project.
        blob_storage_url (str): Optional URL for remote storage (internal use).
        container_name (str): Optional remote storage container name (internal use).
        cache_file (str): Name of the sql file where the cache is stored.
        cache_folder (str): Folder where the cache is stored. Relative to the root folder of the project.
        judge_llm (JudgeLLMConfig): Configuration for the LLM used as a judge in metrics.
    """

    dataset_folder: str
    tokenizer_folder: str
    blob_storage_url: str = ""
    container_name: str = ""
    cache_file: str
    cache_folder: str
    output_folder: str
    resources_folder: str
    retry: RetryConfig
    judge_llm: JudgeLLMConfig
    hallucination_judge_llm: JudgeLLMConfig | None = None
    hallucination: HallucinationConfig = Field(default_factory=HallucinationConfig)
    semscore: SemScoreConfig


def load_config() -> Config:
    """Loads the config and returns ist.

    Returns:
        Config: The config of lm bench.
    """
    config_path = Path(__file__).parent / "config.yaml"
    conf_obj = OmegaConf.to_object(OmegaConf.load(config_path))
    conf = Config(**conf_obj)  # type: ignore[reportCallIssue]
    return conf


CONFIG = load_config()

PROJECT_ROOT = Path(__file__).parent.parent.parent

if df := os.getenv("DATASET_FOLDER"):
    DATASET_FOLDER = Path(df)
else:
    DATASET_FOLDER = PROJECT_ROOT / CONFIG.dataset_folder

if tf := os.getenv("TOKENIZER_FOLDER"):
    TOKENIZER_FOLDER = Path(tf)
else:
    TOKENIZER_FOLDER = PROJECT_ROOT / CONFIG.tokenizer_folder

if cf := os.getenv("CACHE_FOLDER"):
    CACHE_FOLDER = Path(cf)
else:
    CACHE_FOLDER = PROJECT_ROOT / CONFIG.cache_folder

if of := os.getenv("OUTPUT_FOLDER"):
    OUTPUT_FOLDER = Path(of)
else:
    OUTPUT_FOLDER = PROJECT_ROOT / CONFIG.output_folder

if rf := os.getenv("RESOURCES_FOLDER"):
    RESOURCES_FOLDER = Path(rf)
else:
    RESOURCES_FOLDER = PROJECT_ROOT / CONFIG.resources_folder

if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)

if not os.path.exists(TOKENIZER_FOLDER):
    os.makedirs(TOKENIZER_FOLDER)

if not os.path.exists(CACHE_FOLDER):
    os.makedirs(CACHE_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
