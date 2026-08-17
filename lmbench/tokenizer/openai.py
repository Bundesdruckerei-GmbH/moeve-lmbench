"""Tokenizer for OpenAI models."""

from typing import TYPE_CHECKING, override

import tiktoken

if TYPE_CHECKING:
    from lmbench.models.abstract import LLM
    from lmbench.models.data_models import LLMMessage
from lmbench.tokenizer.abstract import Tokenizer


class OpenAITokenizer(Tokenizer):
    """Tokenizer for OpenAI models."""

    TEMPLATE_OVERHEAD = 10

    name: str = "openai"

    @override
    def __init__(self, llm: "LLM"):
        super().__init__(llm)

        # tiktoken's prefix map only matches "gpt-5-", so the gpt-5.x family
        # (gpt-5.1, gpt-5.4-nano, ...) is unmapped. They all use o200k_base.
        # https://github.com/openai/tiktoken/issues/464
        if self.llm.model.startswith("gpt-5"):
            self.tokenizer = tiktoken.get_encoding("o200k_base")
            return

        self.tokenizer = tiktoken.encoding_for_model(self.llm.model)

    @override
    def num_tokens(self, messages: list["LLMMessage"]) -> int:
        text = "\n".join([message.content for message in messages])
        return len(self.tokenizer.encode(text=text)) + (len(messages) * self.TEMPLATE_OVERHEAD)
