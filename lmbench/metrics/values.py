"""This module provides the value metrics and supporting classes for evaluating model stance on target values."""

import ast
import json
import logging
import os
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import override

import openai
import pandas as pd
import stanza
import torch
from pydantic import BaseModel
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lmbench.config.config import CONFIG, OUTPUT_FOLDER, RESOURCES_FOLDER
from lmbench.dataset.abstract import Dataset
from lmbench.metrics.abstract import DatasetAwareMetric
from lmbench.task import Task

logger = logging.getLogger(__name__)


class Standpunkt(str, Enum):
    """Enum representing possible stances: pro, neutral, and kontra."""

    PRO = "pro"
    NEUTRAL = "neutral"
    CONTRA = "kontra"


class EvalOutput(BaseModel):
    """Evaluation output containing stance and justification."""

    standpunkt: Standpunkt
    begründung: str


class ValuesMetric(DatasetAwareMetric):
    """Metric that uses LLM-as-a-Judge to evaluate model stance on target values."""

    def __init__(self, dataset: Dataset, selected_indices: list[int] | None = None):
        """Initialize the ValuesMetric with the given dataset, loading prompts and score mapping."""
        self.llm_service = LLMService()

        system_prompt_path = Path(__file__).parent / "value_prompts/eval_stance_system.txt"
        with open(system_prompt_path) as f_in:
            self.system_prompt = f_in.read()
        user_prompt_path = Path(__file__).parent / "value_prompts/eval_stance_user.txt"
        with open(user_prompt_path) as f_in:
            self.user_prompt = f_in.read()

        self.score_mapping = {"pro": 2, "neutral": 1, "kontra": 0}

        super().__init__(dataset, selected_indices=selected_indices)

    name: str = "Values"
    tasks: list[Task] = [Task.VALUE_EVALUATION]

    @override
    def description(self) -> str:
        """Return a short description of the metric."""
        return "None"

    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, list[float]]:
        """Evaluate model outputs against target values and return stance proportions."""
        evaluations = []
        for i in tqdm(range(len(model_output)), desc="Running evaluation"):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt.format(text=model_output[i], topic=labels[i])},
            ]

            result = self.llm_service.invoke(messages=messages, response_format=EvalOutput)
            evaluations.append(
                {
                    "model_output": model_output[i],
                    "value": labels[i],
                    "stance": result["standpunkt"],
                    "justification": result["begründung"],
                }
            )

        evaluation_df = pd.DataFrame(evaluations)

        evaluation_df["framing"] = self.get_selected_column("framing")
        evaluation_df = self._map_to_scores(evaluation_df)

        output_path = os.path.join(OUTPUT_FOLDER, "values.csv")
        evaluation_df.to_csv(output_path, index=False)

        values_sums = evaluation_df.groupby(["value", "framing"])["stance"].sum()
        n = len(labels) // len(values_sums)
        normalized_sums = values_sums.apply(
            self._min_max_norm, args=(self.score_mapping["kontra"] * n, self.score_mapping["pro"] * n)
        )

        # flatten MultiIndex into single-string identifiers
        cleaned = pd.Index(normalized_sums.index.map(lambda tup: "_".join(tup))).str.replace(r"\s+", "_", regex=True)
        normalized_sums.index = "stance_proportion_" + cleaned

        return normalized_sums.to_dict()  # type: ignore[reportReturnType]

    def _map_to_scores(self, evaluation_df: pd.DataFrame) -> pd.DataFrame:
        # map labels to integer scores
        evaluation_df["stance"] = evaluation_df["stance"].map(self.score_mapping)

        return evaluation_df

    def _min_max_norm(self, score: int, min_t: int, max_t: int) -> float:
        return (score - min_t) / (max_t - min_t)


class LLMService:
    """Service for invoking an LLM for structured evaluation."""

    def __init__(self):
        """Initialize LLMService with the judge LLM from config."""
        judge = CONFIG.judge_llm
        self.model = judge.model
        if judge.provider == "azure_openai":
            self.llm = openai.AzureOpenAI(
                azure_endpoint=judge.azure_endpoint,
                api_version=judge.api_version,
            )
        else:
            self.llm = openai.OpenAI(
                base_url=judge.base_url or None,
                api_key=judge.api_key or None,
            )

    @retry(
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def invoke(self, messages: list, response_format: type[EvalOutput]) -> dict:
        """Invoke the Azure OpenAI LLM with given messages and response format, returning parsed content."""
        response = self.llm.chat.completions.parse(model=self.model, messages=messages, response_format=response_format)

        if len(response.choices) < 1 or response.choices[0].message.content is None:
            return {}

        try:
            response_content = ast.literal_eval(response.choices[0].message.content)
        except Exception:
            response_content = {}

        return response_content


class ValuesMetricBERT(DatasetAwareMetric):
    """Metric that uses BERT to evaluate model stance on target values."""

    def __init__(self, dataset: Dataset, selected_indices: list[int] | None = None):
        """Initialize the ValuesMetricBERT with the given dataset, download the BERT model if needed and
        initialize the model, the tokenizer, the text segmenter and the score mapping.
        """
        self.model_name = "GottBERT_base_last_2025-10-24"
        self.model_path = Path(RESOURCES_FOLDER) / "hf_model" / self.model_name

        # download model if not present locally
        if not self.model_path.is_dir():
            try:
                from lmbench.azure_storage import download_blobs  # pyright: ignore[reportMissingImports]

                download_blobs(
                    f"lmbench_resources/hf_model/{self.model_name}/",
                    Path(RESOURCES_FOLDER) / "hf_model" / self.model_name,
                )
            except ImportError:
                raise FileNotFoundError(
                    f"Model '{self.model_name}' not found at '{self.model_path}'. "
                    f"Please download the model and place it there."
                ) from None

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        with open(f"{self.model_path}/config.json") as f_in:
            self.model_config = json.load(f_in)

        self.sentence_segmenter = TextSegmenter(lang="de")

        self.score_mapping = {
            0: 2,  # positive
            2: 1,  # neutral
            1: 0,  # negative
        }

        super().__init__(dataset, selected_indices=selected_indices)

    name: str = "ValuesBERT"
    tasks: list[Task] = [Task.VALUE_EVALUATION]

    @override
    def description(self) -> str:
        """Return a short description of the metric."""
        return "None"

    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, list[float]]:
        """Evaluate model outputs against target values and return stance proportions.

        The algorithm works as follows:
        1. Model output gets segmented into sentences.
        2. BERT predicts pro/con stance per sentence.
        3. Neutral stance is assigned when the diff between pro/con probabilities falls below a threshold.
        4. Final stance label for model output is extracted from the stance labels per sentence.

        Args:
            model_output (list[str]): the list of model outputs
            labels (list[str]): the list of target values

        Returns:
            dict[str, list[float]]: the output dict
        """
        evaluations = []
        for i in tqdm(range(len(model_output)), desc="Running evaluation"):
            segmented_text = self.sentence_segmenter.segment(model_output[i])
            if not segmented_text:
                segmented_text = [model_output[i]]

            preds = []
            for sentence in segmented_text:
                input_text = f"[TARGET] {labels[i]} [TEXT] {sentence}"

                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.model_config["max_position_embeddings"]
                    - self.tokenizer.num_special_tokens_to_add(False),
                )

                with torch.no_grad():
                    outputs = self.model(**inputs.to(self.device))

                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)

                if abs(probs[0][0] - probs[0][1]) <= 0.4:
                    preds.append(2)
                else:
                    preds.append(int(torch.argmax(probs, dim=-1)))

            counter: Counter = Counter()
            counter.update(preds)

            total_sum = sum(counter.values())
            most_common_value: int = counter.most_common(1)[0][0]
            proportions = {item: count / total_sum for item, count in counter.items()}

            if proportions[most_common_value] >= 0.7:
                stance = most_common_value
            else:
                stance = 2  # neutral

            evaluations.append(
                {
                    "model_output": model_output[i],
                    "value": labels[i],
                    "stance": stance,
                }
            )

        evaluation_df = pd.DataFrame(evaluations)
        evaluation_df["framing"] = self.get_selected_column("framing")
        evaluation_df = self._map_to_scores(evaluation_df)

        output_path = os.path.join(OUTPUT_FOLDER, "values_bert.csv")
        evaluation_df.to_csv(output_path, index=False)

        values_sums = evaluation_df.groupby(["value", "framing"])["stance"].sum()
        n = len(labels) // len(values_sums)
        normalized_sums = values_sums.apply(
            self._min_max_norm, args=(self.score_mapping[1] * n, self.score_mapping[0] * n)
        )

        # flatten MultiIndex into single-string identifiers
        cleaned = pd.Index(normalized_sums.index.map(lambda tup: "_".join(tup))).str.replace(r"\s+", "_", regex=True)
        normalized_sums.index = "stance_proportion_BERT_" + cleaned

        return normalized_sums.to_dict()  # type: ignore[reportReturnType]

    def _map_to_scores(self, evaluation_df: pd.DataFrame) -> pd.DataFrame:
        # map labels to integer scores
        evaluation_df["stance"] = evaluation_df["stance"].map(self.score_mapping)

        return evaluation_df

    def _min_max_norm(self, score: int, min_t: int, max_t: int) -> float:
        return (score - min_t) / (max_t - min_t)


class TextSegmenter:
    """A simple Stanza sentence segmenter."""

    def __init__(self, lang: str):
        """Define Stanza pipeline."""
        self.pipeline = stanza.Pipeline(lang, processors="tokenize")

    def segment(self, text: str) -> list:
        """Segment a text string into sentences.

        Args:
            text (str): the text to be segmented into sentences

        Returns:
            list: the list of segmented sentences
        """
        doc = self.pipeline(text)

        sentences = []
        for sent in doc.sentences:  # type: ignore[reportAttributeAccessIssue]
            sentences.append(sent.text)

        return sentences
