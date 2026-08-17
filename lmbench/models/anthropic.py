"""Implements Claude models using the Anthropic Messages API."""

import logging
import os
from typing import Any, override

import anthropic
import dotenv
from anthropic.types import MessageParam

from lmbench.models.abstract import LLM
from lmbench.models.data_models import LLMMessage, LLMOutput, LLMRole
from lmbench.utils import context_size_eception_from_errormsg

logger = logging.getLogger(__name__)

dotenv.load_dotenv()


class AnthropicLLM(LLM):
    """Implements Claude models using Anthropic's Messages API."""

    def __init__(
        self,
        model: str,
        api_key: str = "ANTHROPIC_API_KEY",
        base_url: str | None = None,
        max_retries: int = 8,
        **kwargs: Any,
    ):
        """Initialize an Anthropic LLM instance.

        Args:
            model (str): The Claude model to use (e.g., "claude-opus-4-7").
            api_key (str): The name of the environment variable holding the API key.
            base_url (str | None): Optional override for the Anthropic API base URL.
            max_retries (int): SDK-level retries that honor the ``retry-after`` header on 429s. The
                outer tenacity wrapper handles longer-term backoff once these are exhausted.
            kwargs (Any): Additional keyword arguments forwarded to ``messages.create``.
        """
        timeout_seconds = kwargs.pop("timeout", 300)
        super().__init__(model=model, **kwargs)

        client_kwargs: dict[str, Any] = {
            "api_key": os.getenv(api_key),
            "timeout": timeout_seconds,
            "max_retries": max_retries,
        }
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self.anthropic_client = anthropic.Anthropic(**client_kwargs)

    @staticmethod
    @override
    def name() -> str:
        return "anthropic"

    @override
    def _invoke(self, messages: list[LLMMessage]) -> str:
        system_parts: list[str] = []
        conv_messages: list[MessageParam] = []
        for m in messages:
            if m.role == LLMRole.SYSTEM:
                system_parts.append(m.content)
            else:
                conv_messages.append({"role": m.role.value, "content": m.content})

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.num_predict,
            "messages": conv_messages,
            **self._kwargs,
        }
        if system_parts:
            create_kwargs["system"] = "\n\n".join(system_parts)
        if self.temperature is not None:
            create_kwargs["temperature"] = self.temperature

        try:
            response = self.anthropic_client.messages.create(**create_kwargs)
        except anthropic.BadRequestError as e:
            raise context_size_eception_from_errormsg(getattr(e, "message", str(e)))

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                reasoning_parts.append(block.thinking)

        text_output = "".join(text_parts)
        if reasoning_parts:
            reasoning = "\n".join(reasoning_parts)
            return LLMOutput(
                result=text_output,
                reasoning_output=reasoning,
                original_output=f"<think>{reasoning}</think>{text_output}",
            )
        if not text_output:
            logger.error(
                "Anthropic completion is empty! stop_reason=%s usage=%s messages=%s",
                response.stop_reason,
                response.usage,
                messages,
            )
            return ""
        return text_output
