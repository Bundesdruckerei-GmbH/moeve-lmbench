import pytest

from lmbench.models.abstract import LLM


@pytest.fixture(autouse=True)
def disable_tenacity_retry(monkeypatch):
    # bypass tenacity retry, use the raw invoke implementation
    monkeypatch.setattr(LLM, "invoke", LLM.invoke.__wrapped__)
