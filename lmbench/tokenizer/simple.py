"""A simple tokenizer that approximates tokens with the formula 4 token == 1 word."""

import os
from typing import override

import dotenv

from lmbench.models.data_models import LLMMessage
from lmbench.tokenizer.abstract import Tokenizer

dotenv.load_dotenv()


class SimpleTokenizer(Tokenizer):
    """A basic tokenizer that approximates token based on the number of words.

    The tokenizer splits the messages into words and approximates that 1 word equals n tokens.
    """

    """TEMPLATE_OVERHEAD accounts for the tokens used to encode the chat template."""
    TEMPLATE_OVERHEAD = 10
    """How many tokens are needed to represent a word."""
    WORD_TO_TOKEN_RATIO: int = int(os.getenv("WORD_TO_TOKEN_RATIO", 4))

    name: str = "simple"

    @override
    def num_tokens(self, messages: list[LLMMessage]) -> int:
        n_tokens = self.TEMPLATE_OVERHEAD * len(messages)  # Add template overhead for each message.
        for msg in messages:
            n_tokens += len(msg.content.split()) * self.WORD_TO_TOKEN_RATIO
        return n_tokens
