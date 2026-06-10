from unittest.mock import MagicMock, patch

import pandas as pd
from langchain_core.outputs import Generation, LLMResult
from ragas.llms.base import BaseRagasLLM
from ragas.metrics import NoiseSensitivity
from ragas.run_config import RunConfig

from lmbench.dataset.dataframe import DataframeDataset, DataframeDatasetConfig
from lmbench.metrics.ragas_metrics import (
    RagasComparisonMetrics,
    RagasQAMetrics,
    RagasTopicExtractionMetrics,
    load_or_adapt_prompts_to_deutsch,
)


def fake_llm():
    class EchoLLM(BaseRagasLLM):
        def generate_text(  # type: ignore
            self,
            prompt,
            *args,  # noqa: ARG002
            **kwargs,  # noqa: ARG002
        ) -> LLMResult:
            return LLMResult(generations=[[Generation(text=prompt.to_string())]])

        async def agenerate_text(  # type: ignore
            self,
            prompt,
            *args,  # noqa: ARG002
            **kwargs,  # noqa: ARG002
        ) -> LLMResult:
            return LLMResult(generations=[[Generation(text=prompt.to_string())]])

        def is_finished(  # type: ignore
            self,
            response,  # noqa: ARG002
            *args,  # noqa: ARG002
            **kwargs,  # noqa: ARG002
        ) -> bool:
            return True

    fake_llm = EchoLLM()
    fake_llm.set_run_config(RunConfig(max_retries=0))

    return fake_llm


def test_load_or_adapt_prompts_to_deutsch():
    metrics = [NoiseSensitivity()]
    load_or_adapt_prompts_to_deutsch(metrics)
    assert (
        "John ist ein fleißiger Student" in metrics[0].get_prompts()["n_l_i_statement_prompt"].examples[0][0].context
    ), "metric prompt should contain german examples"


def test_ragas_comparison_metrics():
    with patch("lmbench.metrics.ragas_metrics.create_llm_wrapper") as mock_create_llm_wrapper:
        mock_create_llm_wrapper.return_value = fake_llm()

        ragas_comparison_metrics = RagasComparisonMetrics()
        result = ragas_comparison_metrics.evaluate(["some output text"], ["some expected text"])

    assert "factual_correctness" in result.keys()


def test_ragas_qa_metrics():
    mock_dataframe_config = DataframeDatasetConfig(
        system_prompt="system_prompt.txt",
        user_prompt="user_prompt.txt",
        target_column="target_column",
        shrink_column="shrink_column",
        context_column="context_column",
        question_column="question_column",
    )

    mock_dataframe = pd.DataFrame(
        {
            "question_column": ["question"],
            "context_column": ["context"],
            "target_column": ["target"],
        }
    )

    MockDataframeDataset = MagicMock(spec=DataframeDataset)
    MockDataframeDataset.config = mock_dataframe_config
    MockDataframeDataset.dataset = mock_dataframe

    with patch("lmbench.metrics.ragas_metrics.create_llm_wrapper") as mock_create_llm_wrapper:
        mock_create_llm_wrapper.return_value = fake_llm()

        ragas_comparison_metrics = RagasQAMetrics(MockDataframeDataset)
        result = ragas_comparison_metrics.evaluate(["some output text"], ["some expected text"])

    assert "faithfulness" in result.keys()
    assert "noise_sensitivity_relevant" in result.keys()


def test_ragas_topic_extraction_metrics():
    with patch("lmbench.metrics.ragas_metrics.create_llm_wrapper") as mock_create_llm_wrapper:
        mock_create_llm_wrapper.return_value = fake_llm()
        ragas_topic = RagasTopicExtractionMetrics()
        result = ragas_topic.evaluate(["economy, healthcare"], ["economy, healthcare"])
    assert "topic_match" in result.keys()
