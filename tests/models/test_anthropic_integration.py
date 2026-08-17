"""Integration tests for the Anthropic provider using real API calls.

These tests are skipped unless ``ANTHROPIC_API_KEY`` is set. Defaults target the
public Anthropic API with ``claude-haiku-4-5`` to keep cost minimal. To run against
an alternate endpoint (e.g. Anthropic models served via Azure AI Foundry), set:

  - ``ANTHROPIC_BASE_URL`` — base URL of the endpoint
  - ``ANTHROPIC_MODEL``    — deployment / model name on that endpoint
"""

from __future__ import annotations

import os

import pytest

from lmbench.models.anthropic import AnthropicLLM
from lmbench.models.data_models import LLMMessage, LLMRole

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - skipping Anthropic integration tests.",
)


def _arg_string(extra: str = "") -> str:
    parts = [extra] if extra else []
    if base_url := os.getenv("ANTHROPIC_BASE_URL"):
        parts.append(f"base_url={base_url}")
    return ",".join(parts)


MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")


def test_invoke_round_trip():
    llm = AnthropicLLM.from_arg_string(model=MODEL, arg_string=_arg_string("num_predict=64"))
    result = llm.invoke(
        [
            LLMMessage(role=LLMRole.SYSTEM, content="Reply with a single word."),
            LLMMessage(role=LLMRole.USER, content="Say hi."),
        ]
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_count_tokens_round_trip():
    llm = AnthropicLLM.from_arg_string(model=MODEL, arg_string=_arg_string("tokenizer=anthropic"))
    n = llm.tokenizer.num_tokens(
        [
            LLMMessage(role=LLMRole.SYSTEM, content="You are concise."),
            LLMMessage(role=LLMRole.USER, content="Hello world."),
        ]
    )
    assert isinstance(n, int)
    assert n > 0
