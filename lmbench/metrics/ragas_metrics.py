"""This module contains two classes RagasQAMetrics and RagasComparisonMetrics that inherit from the Metric class.
RagasQAMetrics is used for evaluating the generation step of question answering systems using some of the metrics
provided by the Ragas library. RagasComparisonMetrics is used for general natural language comparison using some of the
metrics provided by the Ragas library. Both classes use the LangchainLLMWrapper for language model wrapping and
AzureChatOpenAI for interfacing with the Azure OpenAI API. The module also contains helper functions
load_or_adapt_prompts_to_deutsch and create_llm_wrapper for loading or translating prompts to German and creating a
LangchainLLMWrapper instance respectively.
"""

import asyncio  # noqa: I001
import logging
from math import nan
from pathlib import Path
from typing import Any, override

import openai
from ragas import RunConfig, evaluate, DiskCacheBackend, EvaluationDataset, SingleTurnSample, experiment
from ragas.llms import LangchainLLMWrapper, llm_factory
from ragas.metrics import FactualCorrectness, Faithfulness, NoiseSensitivity
from ragas.metrics.collections import FactualCorrectness as FactualCorrectnessV2
from ragas.backends import InMemoryBackend

from datasets import Dataset as arrow_Dataset  # type: ignore
from lmbench.config.config import CONFIG, JudgeLLMConfig, CACHE_FOLDER
from lmbench.dataset.abstract import Dataset
from lmbench.dataset.dataframe import DataframeDataset
from lmbench.metrics import Metric
from lmbench.metrics.abstract import DatasetAwareMetric
from lmbench.task import Task
from lmbench.utils import map_list_to_dict
from lmbench.metrics.llm_judge_topic_extraction import TopicMatch

logger = logging.getLogger(__name__)


def _create_langchain_llm(judge_config: JudgeLLMConfig):
    """Create a LangChain chat model from the judge LLM config.

    Args:
        judge_config: The judge LLM configuration.

    Returns:
        A LangChain chat model instance.
    """
    lm = judge_config.lm_args
    common: dict[str, Any] = {
        "seed": lm.seed if lm.seed is not None else 42,
        "temperature": lm.temperature if lm.temperature is not None else 0,
    }
    if lm.top_p is not None:
        common["top_p"] = lm.top_p
    if lm.max_tokens is not None:
        common["max_tokens"] = lm.max_tokens

    if judge_config.provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(  # type: ignore[reportArgumentType]
            model=judge_config.model,
            api_version=judge_config.api_version,
            azure_deployment=judge_config.azure_deployment,
            azure_endpoint=judge_config.azure_endpoint,
            **common,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=judge_config.model,
            base_url=judge_config.base_url or None,
            api_key=judge_config.api_key or None,  # pyright: ignore[reportArgumentType]
            **common,
        )


def load_or_adapt_prompts_to_deutsch(metrics):
    """Load the german (named 'deutsch' in ragas) prompts or adapt (translate) them if they don't exist.

    Args:
        metrics (list): List of metrics for which prompts need to be adapted.
    """
    adapt_llm = LangchainLLMWrapper(_create_langchain_llm(CONFIG.judge_llm))

    for metric in metrics:
        prompts_path = Path(__file__).parents[0] / "ragas_prompts" / metric.name
        if not prompts_path.exists():
            prompts_path.mkdir(parents=True, exist_ok=True)
        try:
            metric.set_prompts(**metric.load_prompts(prompts_path, "deutsch"))
        except FileNotFoundError:
            logger.info(f"adapting metric {metric} to deutsch")
            adapted_prompts = asyncio.run(metric.adapt_prompts("deutsch", llm=adapt_llm))
            metric.set_prompts(**adapted_prompts)
            metric.save_prompts(prompts_path)


def create_llm_wrapper(judge_config: JudgeLLMConfig):
    """Create a LangchainLLMWrapper instance from the judge LLM config.

    Args:
        judge_config: The judge LLM configuration.

    Returns:
        LangchainLLMWrapper: A LangchainLLMWrapper instance.
    """
    return LangchainLLMWrapper(
        _create_langchain_llm(judge_config),
        cache=DiskCacheBackend(cache_dir=str(CACHE_FOLDER)),
    )


def create_llm_v2(judge_config: JudgeLLMConfig):
    """Create an LLM instance using the new ragas llm_factory API (v0.4+).

    Note: DiskCacheBackend is not used here because the new API returns Pydantic models
    that cannot be pickled by diskcache. Only InMemoryBackend works.

    Args:
        judge_config: The judge LLM configuration.

    Returns:
        InstructorBaseRagasLLM: An LLM instance compatible with ragas.metrics.collections.
    """
    if judge_config.provider == "azure_openai":
        client = openai.AsyncAzureOpenAI(
            api_version=judge_config.api_version,
            azure_endpoint=judge_config.azure_endpoint,
            azure_deployment=judge_config.azure_deployment,
        )
    else:
        client = openai.AsyncOpenAI(
            base_url=judge_config.base_url or None,
            api_key=judge_config.api_key or None,
        )

    lm = judge_config.lm_args
    kwargs: dict[str, Any] = {
        "model": judge_config.model,
        "client": client,
        "seed": lm.seed if lm.seed is not None else 42,
        "temperature": lm.temperature if lm.temperature is not None else 0,
    }
    if lm.top_p is not None:
        kwargs["top_p"] = lm.top_p
    if lm.max_tokens is not None:
        kwargs["max_tokens"] = lm.max_tokens
    return llm_factory(**kwargs)


class RagasQAMetrics(DatasetAwareMetric):
    """
    Class using some of the metrics for RAG systems provided by Ragas for evaluating the generation step.

    Attributes:
        name (str): Name of the metric.
        tasks (list): List of tasks for which the metric is applicable.
    """

    name = "RagasQA"
    tasks: list[Task] = [Task.QUESTION_ANSWERING]

    @override
    def description(self) -> str:
        return (
            "Some of the metrics provided by the ragas library for evaluating the generation step of question"
            " answering systems."
        )

    @override
    def __init__(self, dataset: Dataset, selected_indices: list[int] | None = None):
        super().__init__(dataset=dataset, selected_indices=selected_indices)

        self.llm = create_llm_wrapper(CONFIG.judge_llm)

        self.metrics = [
            NoiseSensitivity(),
            Faithfulness(),
        ]

        load_or_adapt_prompts_to_deutsch(self.metrics)

    @override
    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, list[float]]:
        assert isinstance(self.dataset, DataframeDataset)
        config = self.dataset.config
        eval_dataset = self.dataset.dataset.copy()[
            [config.question_column, config.context_column, config.target_column]
        ]
        eval_dataset.columns = ["user_input", "retrieved_contexts", "reference"]  # type: ignore
        eval_dataset["retrieved_contexts"] = eval_dataset["retrieved_contexts"].apply(lambda x: [x])
        if self.selected_indices is not None:
            eval_dataset = eval_dataset.iloc[self.selected_indices].copy()
        else:
            eval_dataset = eval_dataset.head(len(model_output)).copy()
        eval_dataset["response"] = model_output

        results = evaluate(
            dataset=arrow_Dataset.from_pandas(eval_dataset),
            metrics=self.metrics,
            llm=self.llm,
            run_config=RunConfig(max_wait=CONFIG.retry.max_wait, max_retries=CONFIG.retry.stop_after_attempt),
        )

        scores = map_list_to_dict(results.scores)  # type: ignore[reportAttributeAccessIssue]
        # mapping back to the naming used in previous ragas version
        if "noise_sensitivity(mode=relevant)" in scores:
            scores["noise_sensitivity_relevant"] = scores.pop("noise_sensitivity(mode=relevant)")

        return scores


class RagasComparisonMetrics(Metric):
    """Class using some of the metrics provided by Ragas for general natural language comparison.

    Attributes:
        name (str): Name of the metric.
        tasks (list): List of tasks for which the metric is applicable.
    """

    name = "RagasComparison"
    tasks: list[Task] = [Task.QUESTION_ANSWERING, Task.SUMMARIZATION]

    @override
    def description(self) -> str:
        return "Some of the metrics provided by the ragas library for natural language comparison."

    @override
    def __init__(self):
        super().__init__()

        self.llm = create_llm_wrapper(CONFIG.judge_llm)

        self.metrics = [
            FactualCorrectness(),
        ]

        load_or_adapt_prompts_to_deutsch(self.metrics)

    @override
    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, list[float]]:
        eval_dataset = arrow_Dataset.from_dict({"response": model_output, "reference": labels})

        results = evaluate(
            dataset=eval_dataset,
            metrics=self.metrics,
            llm=self.llm,
            run_config=RunConfig(max_wait=CONFIG.retry.max_wait, max_retries=CONFIG.retry.stop_after_attempt),
        )

        scores = map_list_to_dict(results.scores)  # type: ignore[reportAttributeAccessIssue]

        # mapping back to the naming used in previous ragas version
        if "factual_correctness(mode=f1)" in scores:
            scores["factual_correctness"] = scores.pop("factual_correctness(mode=f1)")

        return scores


class RagasTopicExtractionMetrics(Metric):
    """Class using TopicMatch metric provided by Ragas for topic extraction."""

    name = "RagasTopicExtraction"
    tasks: list[Task] = [Task.TOPIC_EXTRACTION]

    @override
    def description(self) -> str:
        return "Topic extraction evaluation using the TopicMatch metric via Ragas."

    @override
    def __init__(self):
        super().__init__()

        self.llm = create_llm_wrapper(CONFIG.judge_llm)

        self.topicmatch = TopicMatch()
        self.metrics = [
            self.topicmatch,
        ]

        load_or_adapt_prompts_to_deutsch(self.metrics)

    def words_factor(self, reference_topics: list[str], response_topics: list[str]) -> float:
        """Calculates a word factor that estimates how much more words the response topics contain than the reference
        topics. If the reference is longer than the response topics, the resulting factor is clamped at 1.0.

        Args:
            reference_topics (list[str]): list of reference topics
            response_topics (list[str]): list of response topics

        Returns:
            float: The factor by which the response topics have more words than the reference topics. At least 1.0.
        """
        response_words = sum([len(t.split()) for t in response_topics])
        reference_words = sum([len(t.split()) for t in reference_topics])

        if reference_words == 0:
            return nan

        return max(1.0, response_words / reference_words)

    def topics_factor(self, reference_topics: list[str], response_topics: list[str]) -> float:
        """Calculate a topic factor that measures how large the absolute difference between reference and response
        topics is.

        Here a factor of 1.0 means there are the same number of topics in the reference and the response. A factor of
        2.0 either means that there are twice as many response topics than the reference topics or that there are half
        as many.

        Args:
            reference_topics (list[str]): List of reference topics.
            response_topics (list[str]): List of response topics.

        Returns:
            float: The factor of the  absolute difference between the number of reference and response topics
        """
        abs_diff = abs(len(reference_topics) - len(response_topics))
        return 1 + (abs_diff / len(reference_topics)) if reference_topics else nan

    @override
    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, list[float]]:
        eval_dataset = arrow_Dataset.from_dict({"response": model_output, "reference": labels})

        results = evaluate(
            dataset=eval_dataset,
            metrics=self.metrics,
            llm=self.llm,
            run_config=RunConfig(max_wait=CONFIG.retry.max_wait, max_retries=CONFIG.retry.stop_after_attempt),
        )

        scores = map_list_to_dict(results.scores)  # type: ignore[reportAttributeAccessIssue]

        if "topic_match(mode=f1)" in scores:
            scores["topic_match"] = scores.pop("topic_match(mode=f1)")

        words_factors = []
        topics_factors = []
        str_ad = []
        for i, (response, reference) in enumerate(zip(model_output, labels)):
            response_topics = self.topicmatch.topic_cache.get(response, None)
            reference_topics = self.topicmatch.topic_cache.get(reference, None)
            topic_match = scores["topic_match"][i]

            words_factor = nan
            topics_factor = nan
            if response_topics is not None and reference_topics is not None:
                words_factor = self.words_factor(response_topics=response_topics, reference_topics=reference_topics)
                topics_factor = self.topics_factor(response_topics=response_topics, reference_topics=reference_topics)
            words_factors.append(words_factor)
            topics_factors.append(topics_factor)

            structural_adherence = topic_match * (1 / (words_factor)) * (1 / topics_factor)
            str_ad.append(structural_adherence)

        scores["words_factor"] = words_factors
        scores["topic_differnce"] = topics_factors
        scores["structural_adherence"] = str_ad

        return scores


class RagasComparisonMetricsV2(Metric):
    """EXPERIMENTAL: V2 implementation using the new ragas 0.4+ API.

    This version uses:
    - llm_factory instead of deprecated LangchainLLMWrapper
    - ragas.metrics.collections.FactualCorrectness instead of ragas.metrics
    - Direct async metric.ascore() via @experiment decorator

    Known limitations (as of ragas 0.4.2):
    - No disk caching: DiskCacheBackend fails due to pickle issues with Pydantic models
      returned by the new API. Only InMemoryBackend works.
    - No language adaptation: FactualCorrectness uses hardcoded English prompts via
      claim_decomposition_prompt() instead of the standard BasePrompt.examples pattern,
      so prompt.adapt() does not work.
    - No save/load prompts: Not yet implemented for BasePrompt in v0.4.x.

    For production use with German language support and caching, use RagasComparisonMetrics
    (V1 API) instead.

    Attributes:
        name (str): Name of the metric.
        tasks (list): List of tasks for which the metric is applicable.
    """

    name = "RagasComparisonV2"
    tasks: list[Task] = [Task.QUESTION_ANSWERING, Task.SUMMARIZATION]

    @override
    def description(self) -> str:
        return "Ragas factual correctness using the new v0.4+ API (llm_factory + direct scoring)."

    @override
    def __init__(self):
        super().__init__()

        self.llm = create_llm_v2(CONFIG.judge_llm)
        self.factual_correctness = FactualCorrectnessV2(llm=self.llm)

    @override
    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, list[float]]:
        # Create ragas dataset from response/reference pairs
        samples = [
            SingleTurnSample(response=response, reference=reference)
            for response, reference in zip(model_output, labels)
        ]
        dataset = EvaluationDataset(samples=samples)  # type: ignore[reportArgumentType]

        @experiment(backend=InMemoryBackend())
        async def score_factual_correctness(row: SingleTurnSample):
            result = await self.factual_correctness.ascore(  # type: ignore[reportArgumentType]
                response=row.response,  # type: ignore[reportArgumentType]
                reference=row.reference,  # type: ignore[reportArgumentType]
            )
            return {"factual_correctness": result.value}

        # Run experiment over all samples
        experiment_result = asyncio.run(
            score_factual_correctness.arun(dataset)  # type: ignore[reportArgumentType]
        )

        # Convert to dict[str, list[float]] format expected by the Metric interface
        return {"factual_correctness": experiment_result.to_pandas()["factual_correctness"].tolist()}
