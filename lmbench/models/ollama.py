"""Implements LLM using Ollama."""

import logging
from typing import Any, Literal, override

import ollama
from ollama import Message, Options

from lmbench.models.abstract import LLM
from lmbench.models.data_models import LLMMessage, LLMOutput

logger = logging.getLogger(__name__)


class OllamaLLM(LLM):
    """Implements LLM using Ollama."""

    def __init__(
        self,
        model: str,
        host: str = "http://127.0.0.1:11434",
        **kwargs: Any,
    ):
        """
        Initializes the model using Ollama as a backend.

        Args:
            model(str): The name of the model.
            host (str, optional): The host of Ollama to use. Defaults to "http://127.0.0.1:11434".
            **kwargs: Optional arguments to pass to the ollama client.

        """
        super().__init__(model=model, **kwargs)

        self.ollama_client = ollama.Client(host=host)
        self._is_first_invoke = True

        # build options, only include temperature when it's not None
        options_kwargs: dict[str, Any] = {
            "num_ctx": self.ctx_size,
            "num_predict": self.num_predict,
        }
        if self.temperature is not None:
            options_kwargs["temperature"] = self.temperature
        self.options = Options(**options_kwargs)

        self.think: bool | Literal["low", "medium", "high"] = False
        if self._kwargs.get("think"):
            self.think = self._kwargs.pop("think")

        for key, value in self._kwargs.items():
            self.options[key] = value  # type: ignore

    @staticmethod
    @override
    def name() -> str:
        return "ollama"

    @override
    def _invoke(self, messages: list[LLMMessage]) -> str:
        first_invoke = self._is_first_invoke
        if first_invoke:
            logger.info("Checking available Ollama models.")
            available_models = self.ollama_client.list()
            if self.model not in [model.model for model in available_models.models]:
                logger.info(f"Downloading LLM model {self.model} if not available...")
                self.ollama_client.pull(self.model)
                logger.info("Finished downloading.")
            logger.info(f"Sending first request to Ollama for {self.model}; the model may take a moment to load.")
            self._is_first_invoke = False

        ollama_messages = [Message(role=m.role.value, content=m.content) for m in messages]
        response = self.ollama_client.chat(
            model=self.model,
            messages=ollama_messages,
            think=self.think,
            options=self.options,
        )
        if first_invoke:
            logger.info(f"Ollama returned the first response for model {self.model}.")

        if response is None or response.get("message") is None or response["message"].get("content") is None:
            logger.error("Ollama completion response is empty!", messages)
            return ""

        # If reasoning content is available (called "thinking" in ollama), we return an LLMOutput object and
        # "reconstruct" a fake original_output with <think>-tags. We need this to save the full output in the cache
        # later, not just the non-reasoning output.
        if hasattr(response["message"], "thinking") and response["message"].thinking is not None:
            return LLMOutput(
                result=response["message"]["content"],
                reasoning_output=response["message"]["thinking"],
                original_output=f"<think>{response['message']['thinking']}</think>{response['message']['content']}",
            )
        return response["message"]["content"]
