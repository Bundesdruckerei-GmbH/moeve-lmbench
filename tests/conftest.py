import os

import pytest

from lmbench.models.abstract import LLM


@pytest.fixture(autouse=True)
def disable_tenacity_retry(monkeypatch):
    # bypass tenacity retry, use the raw invoke implementation
    monkeypatch.setattr(LLM, "invoke", LLM.invoke.__wrapped__)


@pytest.fixture(autouse=True)
def placeholder_openai_key(monkeypatch):
    # the unit tests mock every LLM call, but the openai SDK still refuses to construct a client
    # without a key. only set a placeholder, never override a real one.
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
