"""Implements GPT models using Azure OpenAI and all OpenAI API compatible LLM endpoints."""

import logging
import os
import typing
from collections.abc import Iterable
from typing import Any, override

import dotenv
import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from lmbench.models.abstract import LLM
from lmbench.models.data_models import LLMMessage, LLMOutput
from lmbench.utils import context_size_eception_from_errormsg

logger = logging.getLogger(__name__)

dotenv.load_dotenv()


def message_to_llmoutput(result: ChatCompletion) -> str:
    """Extracts the reasoning output from the message if it is available.

    Returns either a "normal" string or a string of type LLMOutput with the reasoning_output and original_output set.
    The LLMOutput has a "reconstructed" original_message with <think>-tags. This allows us to save the complete string
    in the cache and - during a cache hit - to reparse the reasoning output.

    Args:
        result (ChatCompletion): A result of an OpenAI chat completion call.

    Returns:
        str: The answer of the llm with reasoning added, if available (as an LLMOutput)
    """
    message = result.choices[0].message
    llm_answer = message.content or ""
    reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
    if reasoning:
        return LLMOutput(
            result=llm_answer, reasoning_output=reasoning, original_output=f"<think>{reasoning}</think>{llm_answer}"
        )
    if result.usage is not None and result.usage.completion_tokens_details is not None:
        n_reasoning_tokens = result.usage.completion_tokens_details.reasoning_tokens or 0
        if n_reasoning_tokens > 0:
            reasoning = f"num_reasoning_tokens:{n_reasoning_tokens}"
            logger.debug(f"Found OpenAI hidden reasoning tokens: {n_reasoning_tokens}")
            return LLMOutput(
                result=llm_answer, reasoning_output=reasoning, original_output=f"<think>{reasoning}</think>{llm_answer}"
            )
    return llm_answer


def _create_chat_completion_with_token_fallback(
    client: Any,
    create_kwargs: dict[str, Any],
    num_predict: int,
) -> tuple[ChatCompletion, bool]:
    """Call chat.completions.create, retrying with max_tokens if max_completion_tokens is rejected.

    Args:
        client: OpenAI or AzureOpenAI client instance.
        create_kwargs: Arguments to forward to `chat.completions.create`.
        num_predict: Desired completion budget to use for the fallback max_tokens.

    Returns:
        A tuple of (ChatCompletion response, should_disable_max_completion_tokens).
        The boolean is True only when the request failed because max_completion_tokens is not accepted,
        indicating the caller should switch to max_tokens for future calls.
    """
    disable_flag = False

    try:
        return client.chat.completions.create(**create_kwargs), disable_flag
    except (openai.UnprocessableEntityError, openai.BadRequestError) as e:
        error_msg = getattr(e, "message", str(e))
        if "max_completion_tokens" not in error_msg and "extra-parameters" not in error_msg:
            raise

        # Signal that max_completion_tokens should be disabled for future calls.
        disable_flag = True

        fallback_kwargs = {**create_kwargs}
        fallback_kwargs.pop("max_completion_tokens", None)
        fallback_kwargs.setdefault("max_tokens", num_predict)
        logger.debug("Retrying with max_tokens because max_completion_tokens was rejected: %s", error_msg)
        return client.chat.completions.create(**fallback_kwargs), disable_flag


class OpenAILLM(LLM):
    """Implements GPT models using OpenAI."""

    def __init__(self, model: str, base_url: str, api_key: str, **kwargs):
        """Initialize an OpenAI API compatible LLM instance with the given model, base_url (without "/v1") and api key.

        Args:
            model (str): The model to use (e.g., "mistral-large")
            base_url (str): The base url to an OpenAI API compatible LLM server
            api_key (str): The enviroment variable name for the API key.
            kwargs (Any): Additional keyword arguments to pass to the model.
        """
        timeout_seconds = kwargs.pop("timeout", 300)
        super().__init__(model=model, **kwargs)

        self.openai_client = openai.Client(
            base_url=base_url,
            api_key=os.getenv(api_key),
            timeout=timeout_seconds,
        )
        self._supports_max_completion_tokens = True

    @staticmethod
    def name() -> str:
        """Returns the name of the provider."""
        return "openai"

    def _invoke(self, messages: list[LLMMessage]) -> str:
        """Invokes the selected model with the OpenAI model and the chat endpoint. Messages given are parsed.

        Args:
            messages (list[LLMMessage]): The messages to be given to the LLM

        Returns:
            str: The output of the LLM
        """
        oai_messages = [m.model_dump() for m in messages]
        typed_messages = typing.cast(Iterable[ChatCompletionMessageParam], oai_messages)
        try:
            # build kwargs, only include temperature when it's not None
            create_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": typed_messages,
                **self._kwargs,
            }
            if self._supports_max_completion_tokens:
                create_kwargs["max_completion_tokens"] = self.num_predict
            else:
                create_kwargs["max_tokens"] = self.num_predict
            if self.temperature is not None:
                create_kwargs["temperature"] = self.temperature

            response, should_disable_max_completion_tokens = _create_chat_completion_with_token_fallback(
                self.openai_client,
                create_kwargs=create_kwargs,
                num_predict=self.num_predict,
            )
            if should_disable_max_completion_tokens:
                self._supports_max_completion_tokens = False
            if len(response.choices) < 1 or response.choices[0].message.content is None:
                logger.error("OpenAI Completion is empty!", messages)
                return ""

            return message_to_llmoutput(response)
        except openai.BadRequestError as e:
            raise context_size_eception_from_errormsg(e.message)


class AzureOpenAILLM(LLM):
    """Implements GPT models using AzureOpenAI."""

    def __init__(self, model: str, azure_endpoint: str, api_version: str, api_key: str, **kwargs: Any):
        """Initialize an Azure OpenAI LLM instance with the given model, azure_endpoint, api version and api key.

        Args:
            model (str): The model to use (e.g., "gpt-4o")
            azure_endpoint (str): The azure endpoint to use
            api_version (str): The Azure API version to use
            api_key (str): The environment variable which stores the api_key.
            kwargs (dict[str, Any]): Additionnal parameters for the model configuration.
        """
        timeout_seconds = kwargs.pop("timeout", 300)
        super().__init__(model=model, **kwargs)

        self.openai_client = openai.AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=os.getenv(api_key),
            api_version=api_version,
            timeout=timeout_seconds,
        )
        self._supports_max_completion_tokens = True

    @staticmethod
    @override
    def name() -> str:
        return "azure_openai"

    @override
    def _invoke(self, messages: list[LLMMessage]) -> str:
        oai_messages = [m.model_dump() for m in messages]
        typed_messages = typing.cast(Iterable[ChatCompletionMessageParam], oai_messages)
        try:
            # build kwargs, only include temperature when it's not None
            create_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": typed_messages,
                **self._kwargs,
            }
            if self._supports_max_completion_tokens:
                create_kwargs["max_completion_tokens"] = self.num_predict
            else:
                create_kwargs["max_tokens"] = self.num_predict
            if self.temperature is not None:
                create_kwargs["temperature"] = self.temperature

            response, should_disable_max_completion_tokens = _create_chat_completion_with_token_fallback(
                self.openai_client,
                create_kwargs=create_kwargs,
                num_predict=self.num_predict,
            )
            if should_disable_max_completion_tokens:
                self._supports_max_completion_tokens = False

            if len(response.choices) < 1 or response.choices[0].message.content is None:
                logger.error("OpenAI Completion is empty!", messages)
                return ""

            return message_to_llmoutput(response)
        except openai.BadRequestError as e:
            raise context_size_eception_from_errormsg(e.message)
