import asyncio
from math import isnan, nan
from unittest.mock import MagicMock, patch

import pandas as pd
from langchain_core.outputs import Generation, LLMResult
from ragas.llms.base import BaseRagasLLM
from ragas.metrics import NoiseSensitivity
from ragas.metrics._faithfulness import StatementGeneratorOutput
from ragas.run_config import RunConfig

from lmbench.dataset.dataframe import DataframeDataset, DataframeDatasetConfig
from lmbench.metrics.ragas_metrics import (
    RagasComparisonMetrics,
    RagasQAMetrics,
    RagasTopicExtractionMetrics,
    _count_judge_failures,
    _FaithfulnessTracked,
    _NoiseSensitivityTracked,
    _relabel_model_broken,
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
    assert "factual_correctness_judge_failure_count" in result.keys()


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
    assert "faithfulness_judge_failure_count" in result.keys()
    assert "noise_sensitivity_relevant_judge_failure_count" in result.keys()


def test_ragas_topic_extraction_metrics():
    with patch("lmbench.metrics.ragas_metrics.create_llm_wrapper") as mock_create_llm_wrapper:
        mock_create_llm_wrapper.return_value = fake_llm()
        ragas_topic = RagasTopicExtractionMetrics()
        result = ragas_topic.evaluate(["economy, healthcare"], ["economy, healthcare"])
    assert "topic_match" in result.keys()
    assert "topic_match_judge_failure_count" in result.keys()


def test_count_judge_failures():
    scores = {
        "faithfulness": [0.9, nan, 0.5, nan],
        "noise_sensitivity_relevant": [nan, 0.7, 0.8, 0.6],
        "unrelated": [1.0, 1.0],
    }

    counts = _count_judge_failures(scores, ["faithfulness", "noise_sensitivity_relevant", "missing_key"])

    assert counts == {
        "faithfulness_judge_failure_count": 2,
        "noise_sensitivity_relevant_judge_failure_count": 1,
    }


def test_relabel_model_broken():
    scores = {
        "faithfulness": [0.9, nan, nan, 0.0],
    }
    # row 1: usable answer -> stays NaN (judge failure, dropped under OMIT)
    # row 2: empty answer -> relabelled to 0.0 (model's fault)
    model_output = ["a fine answer", "another fine answer", "   ", "yet another answer"]

    _relabel_model_broken(scores, ["faithfulness", "missing_key"], model_output)

    assert scores["faithfulness"][0] == 0.9
    assert isnan(scores["faithfulness"][1])
    assert scores["faithfulness"][2] == 0.0
    assert scores["faithfulness"][3] == 0.0


def test_relabel_uses_empty_answer_texts():
    scores = {
        "faithfulness": [nan, nan, 0.5],
    }
    # row 0: judge decomposed the answer into zero statements -> model-broken -> 0.0
    # row 1: usable answer, judge itself failed -> stays NaN (dropped under OMIT)
    model_output = ["word salad blah blah", "a fine answer", "another answer"]

    _relabel_model_broken(scores, ["faithfulness"], model_output, {"word salad blah blah"})

    assert scores["faithfulness"][0] == 0.0
    assert isnan(scores["faithfulness"][1])
    assert scores["faithfulness"][2] == 0.5


def test_tracked_metrics_record_empty_answers():
    async def empty(*_args, **_kwargs):
        return StatementGeneratorOutput(statements=[])

    async def nonempty(*_args, **_kwargs):
        return StatementGeneratorOutput(statements=["a statement"])

    faith = _FaithfulnessTracked(llm=MagicMock())
    faith.statement_generator_prompt.generate = empty  # type: ignore[method-assign]
    asyncio.run(faith._create_statements({"response": "garbage", "user_input": "q"}, None))
    assert "garbage" in faith.empty_answer_texts

    faith_ok = _FaithfulnessTracked(llm=MagicMock())
    faith_ok.statement_generator_prompt.generate = nonempty  # type: ignore[method-assign]
    asyncio.run(faith_ok._create_statements({"response": "a real answer", "user_input": "q"}, None))
    assert faith_ok.empty_answer_texts == set()

    noise = _NoiseSensitivityTracked(llm=MagicMock())
    noise.statement_generator_prompt.generate = empty  # type: ignore[method-assign]
    result = asyncio.run(noise._decompose_answer_into_statements("garbage", "q", None))
    assert result == []
    assert "garbage" in noise.empty_answer_texts


def test_relabel_is_a_noop_on_factual_correctness():
    # factual_correctness floors broken output to 0.0 (never NaN), so relabel changes no value.
    scores = {"factual_correctness": [0.7, 0.0, 0.0]}
    before = list(scores["factual_correctness"])

    _relabel_model_broken(scores, ["factual_correctness"], ["a real answer", "", "another answer"])

    assert scores["factual_correctness"] == before
