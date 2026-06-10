import threading

import pytest

from lmbench.models.abstract import CONFIG, LLM
from lmbench.models.data_models import LLMMessage, LLMOutput, LLMRole


class DummyLLM(LLM):
    """A fake LLM that just counts how often `_invoke` is called."""

    @staticmethod
    def name() -> str:
        return "dummy"

    def __init__(self, model: str, **kwargs):
        # ensure we use a temp cache file
        super().__init__(model=model, **kwargs)
        self.calls = 0

    def _invoke(self, messages: list[LLMMessage]) -> str:  # noqa: ARG002
        self.calls += 1
        # return an LLMOutput with both reasoning and original
        return LLMOutput(
            "generated-text",
            reasoning_output="thinking…",
            original_output="<think>thinking…</think>generated-text",
        )


@pytest.fixture(autouse=True)
def tmp_cache_folder(tmp_path, monkeypatch):
    """Force all DummyLLM instances to use a fresh sqlite file in tmp_path."""
    monkeypatch.setattr("lmbench.models.abstract.CACHE_FOLDER", tmp_path)
    # also override the file name if needed
    monkeypatch.setattr(
        "lmbench.models.abstract.CONFIG",
        CONFIG.__class__(**{**CONFIG.model_dump(), "cache_file": "cache_v2.sqlite"}),
    )
    return tmp_path


def make_message() -> list[LLMMessage]:
    return [LLMMessage(role=LLMRole.USER, content="hello")]


def test_cached_invoke_caches_llmoutput():
    llm = DummyLLM("any")
    msgs = make_message()

    # first call: should invoke once and return an LLMOutput
    out1 = llm.cached_invoke(msgs)
    assert isinstance(out1, LLMOutput)
    assert out1.original_output == "<think>thinking…</think>generated-text"
    assert out1.reasoning_output == "thinking…"
    assert out1 == "generated-text"
    assert llm.calls == 1

    # second call: should hit cache, same type, no extra invoke
    out2 = llm.cached_invoke(msgs)
    assert isinstance(out2, LLMOutput)
    assert out2.original_output == out1.original_output
    assert out2.reasoning_output == out1.reasoning_output
    assert out2 == "generated-text"
    assert llm.calls == 1  # still one invoke_

    # ensure persistence across new instance
    _lock2 = threading.Lock()
    llm2 = DummyLLM("any")
    # the hash is the same, so it will pick up the prior entry
    out3 = llm2.cached_invoke(msgs)
    assert isinstance(out3, LLMOutput)
    assert out3.original_output == out1.original_output
    assert out3.reasoning_output == out1.reasoning_output
    assert out3 == "generated-text"
    assert llm2.calls == 0  # because from cache


class DummyStrLLM(LLM):
    """Fake LLM that returns a raw str instead of LLMOutput."""

    @staticmethod
    def name() -> str:
        return "dummy_str"

    def __init__(self, model: str, **kwargs):
        super().__init__(model=model, **kwargs)
        self.calls = 0

    def _invoke(self, messages: list[LLMMessage]) -> str:  # noqa: ARG002
        self.calls += 1
        return "plain-text"


def test_cached_invoke_caches_str_return():
    llm = DummyStrLLM("any")
    msgs = make_message()

    # first call
    out1 = llm.cached_invoke(msgs)
    assert isinstance(out1, str)
    assert out1 == "plain-text"
    assert llm.calls == 1

    # second call should still return str and not invoke again
    out2 = llm.cached_invoke(msgs)
    assert isinstance(out2, str)
    assert out2 == "plain-text"
    assert llm.calls == 1

    # new instance hits the same cache, so no new invoke
    llm2 = DummyStrLLM("any")
    out3 = llm2.cached_invoke(msgs)
    assert isinstance(out3, str)
    assert out3 == "plain-text"
    assert llm2.calls == 0
