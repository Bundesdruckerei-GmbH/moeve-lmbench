"""
Integration tests for the TopicMatch metric that rely on real LLM calls.

The tests are skipped automatically if no judge LLM is configured.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from ragas import SingleTurnSample

from lmbench.config.config import CONFIG
from lmbench.metrics.llm_judge_topic_extraction import TopicMatch
from lmbench.metrics.ragas_metrics import create_llm_wrapper


def _judge_llm_available() -> bool:
    """Return True if the judge LLM config looks usable."""
    judge = CONFIG.judge_llm
    if judge.provider == "azure_openai":
        return bool(judge.azure_endpoint)
    # For openai provider, check if the env var actually has a value
    return bool(judge.api_key and os.getenv(judge.api_key))


pytestmark = pytest.mark.skipif(
    not _judge_llm_available(),
    reason="Judge LLM not configured – skipping integration tests.",
)


def _judge_llm():
    """Create a judge LLM wrapper from the config."""
    return create_llm_wrapper(CONFIG.judge_llm)


@pytest.mark.asyncio
async def test_topic_match_perfect_match():
    "Identical topic lists should yield an F1 score of 1.0."
    metric = TopicMatch(llm=_judge_llm())

    reference = "economy, healthcare, education"
    response = "education, healthcare, economy"

    sample = SingleTurnSample(reference=reference, response=response)

    score = await metric.single_turn_ascore(sample)

    assert np.isclose(score, 1.0, atol=0.05)


@pytest.mark.asyncio
async def test_topic_match_partial_match():
    "Partially overlapping topics should yield an F1 score smaller than 1.0."
    metric = TopicMatch(llm=_judge_llm())

    reference = "economy, healthcare, education"
    response = "healthcare, sports"

    sample = SingleTurnSample(reference=reference, response=response)

    score = await metric.single_turn_ascore(sample)

    assert np.isclose(score, 0.4, atol=0.05)


@pytest.mark.asyncio
async def test_topic_match_bullet_list_perfect_match():
    "Bullet-list formatted response matching reference should yield an F1 score of 1.0."
    metric = TopicMatch(llm=_judge_llm())

    reference = "economy, healthcare, education"
    response = "* economy\n* healthcare\n* education"

    sample = SingleTurnSample(reference=reference, response=response)

    score = await metric.single_turn_ascore(sample)

    assert np.isclose(score, 1.0, atol=0.05)
