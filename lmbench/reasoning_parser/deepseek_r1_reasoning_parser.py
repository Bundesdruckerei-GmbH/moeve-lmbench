"""Reasoning parser for DeepSeek R1 model."""

import logging
from typing import override

from lmbench.models.data_models import LLMOutput
from lmbench.reasoning_parser.abstract import ReasoningParser

logger = logging.getLogger(__name__)


class DeepSeekR1ReasoningParser(ReasoningParser):
    """
    Reasoning parser for DeepSeek R1 model.
    see https://github.com/vllm-project/vllm/blob/7a8987dac5f0ed0c798a73e8b4ec8f5e640bc63a/vllm/entrypoints/openai/reasoning_parsers/deepseek_r1_reasoning_parser.py.

    The DeepSeek R1 model uses <think>...</think> tokens to denote reasoning
    text. This parser extracts the reasoning content from the model output.
    """

    name: str = "deepseek_r1"

    def __init__(self):
        """Constructor."""
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"

    @override
    def extract_reasoning_content(self, model_output: str) -> str:
        """
        Extract reasoning content from a complete model-generated string.

        Used for non-streaming responses where we have the entire model response
        available before sending to the client.

        Parameters:
        model_output: str
            The model-generated string to extract reasoning content from.

        Returns:
           str: The model output with the reasoning content extracted.
        """
        return LLMOutput.from_string(
            model_output, think_start_token=self.think_start_token, think_end_token=self.think_end_token
        )
