"""Module containing tokenizers."""

from lmbench.tokenizer.abstract import Tokenizer
from lmbench.tokenizer.anthropic import AnthropicTokenizer
from lmbench.tokenizer.huggingface import HuggingfaceTokenizer
from lmbench.tokenizer.openai import OpenAITokenizer
from lmbench.tokenizer.simple import SimpleTokenizer

TOKENIZER_LIST: list[type[Tokenizer]] = [SimpleTokenizer, OpenAITokenizer, HuggingfaceTokenizer, AnthropicTokenizer]
TOKENIZER_MAP: dict[str, type[Tokenizer]] = {tokenizer.name: tokenizer for tokenizer in TOKENIZER_LIST}  # type: ignore
