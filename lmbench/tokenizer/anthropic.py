"""Tokenizer for Anthropic Claude models using the server-side count_tokens endpoint."""

import logging
from typing import TYPE_CHECKING, Any, override

from anthropic.types import MessageParam
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_random_exponential

from lmbench.config.config import CONFIG
from lmbench.models.data_models import LLMRole
from lmbench.tokenizer.abstract import Tokenizer

if TYPE_CHECKING:
    from lmbench.models.data_models import LLMMessage

logger = logging.getLogger(__name__)


class AnthropicTokenizer(Tokenizer):
    """Tokenizer for Claude models.

    Uses Anthropic's ``messages.count_tokens`` endpoint, which is the canonical token
    count for a given model. The endpoint is invoked through the ``AnthropicLLM``'s
    configured client so it inherits the same ``api_key``/``base_url``/``timeout``.
    """

    name: str = "anthropic"

    @retry(
        before_sleep=before_sleep_log(logger, logging.INFO),
        stop=stop_after_attempt(CONFIG.retry.stop_after_attempt),
        wait=wait_random_exponential(min=CONFIG.retry.min_wait, max=CONFIG.retry.max_wait),
    )
    @override
    def num_tokens(self, messages: list["LLMMessage"]) -> int:
        system_parts: list[str] = []
        conv_messages: list[MessageParam] = []
        for m in messages:
            if not m.content or not m.content.strip():
                continue
            if m.role == LLMRole.SYSTEM:
                system_parts.append(m.content)
            else:
                conv_messages.append({"role": m.role.value, "content": m.content})

        if not conv_messages:
            return 0

        kwargs: dict[str, Any] = {"model": self.llm.model, "messages": conv_messages}
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        result = self.llm.anthropic_client.messages.count_tokens(**kwargs)  # type: ignore[attr-defined]
        return result.input_tokens
