"""Tokenizer based on huggingface AutoTokenizer."""

import logging
import os
from typing import TYPE_CHECKING, override

from jinja2 import TemplateError
from transformers import AutoTokenizer

from lmbench.config.config import TOKENIZER_FOLDER
from lmbench.utils import sanitize_name

if TYPE_CHECKING:
    from lmbench.models.abstract import LLM
    from lmbench.models.data_models import LLMMessage

from lmbench.tokenizer.abstract import Tokenizer

logger = logging.getLogger(__name__)


class HuggingfaceTokenizer(Tokenizer):
    """A tokenizer that uses `AutoTokenizer` from `transformers`.

    The tokenizer and config is loaded from the azure blob storage if the corresponding folder in the `TOKENIZER_FOLDER`
    does not exist.

    The tokenizer name will be the longest matching substring (as split by "`-`") of the model name. This means one
    tokenizer can be used for a class of models, e.g., the tokenizer `llama-3.1` can be used for all models in this
    class.

    Attributes:
        tokenizer_name (str): The name of the tokenizer. This may be different from the model name.
        tokenizer (AutoTokenizer): The tokenizer object from `transformers`.
    """

    name: str = "huggingface"

    def _local_tokenizer_name(self, model_name: str) -> str | None:
        """Check for a local tokenizer that matches the given model name or prefix.

        While the tokenizer name may be the same as the model name, many model families use the same tokenizer. That is
        why the model name is split by dash ("-") and substrings for the tokenizer are checked. For example, for the
        model `llama-3.1-70b-instruct`, the following names would be checked:
            - `llama-3.1-70b-instruct`
            - `llama-3.1-70b`
            - `llama-3.1`
            - `llama`

        With this method, all models of the llama-3.1 family can share the same tokenizer.

        If a tokenizer is found, it's name will be returned. If no tokenizer is found, `None` will be returned and the
        tokenizer can be downloaded from the Azure blob storage via the `_load_tokenizer` method.

        Args:
            model_name (str): The name of the model for which the tokenizer name should be extracted.

        Returns:
            str | None: The name of the tokenizer or None if the tokenizer is not found locally.
        """
        model_names = model_name.split("-")
        for i in range(len(model_names)):
            test_name = "-".join(model_names[: len(model_names) - i])
            if os.path.exists(f"{TOKENIZER_FOLDER}/{test_name}"):
                return test_name
        return None

    def _load_tokenizer(self, model_name: str) -> str:
        """Load the tokenizer from remote storage and return the name of the tokenizer.

        Requires the `lmbench.azure` module to be available (internal only).

        Args:
            model_name (str): The name of the model for which the tokenizer should be downloaded.

        Raises:
            ValueError: Raised when the tokenizer can not be found.

        Returns:
            str: The name of the tokenizer.
        """
        try:
            from lmbench.azure_storage import download_tokenizer  # pyright: ignore[reportMissingImports]
        except ImportError:
            raise ValueError(
                f"Tokenizer not found locally for model '{model_name}'. "
                f"Place the tokenizer files in '{TOKENIZER_FOLDER}/{model_name}/'."
            ) from None

        return download_tokenizer(model_name)

    @override
    def __init__(self, llm: "LLM"):
        super().__init__(llm)
        llm_model = sanitize_name(self.llm.model.split("/")[-1]).lower()
        self.tokenizer_name = self._local_tokenizer_name(llm_model)
        if self.tokenizer_name is None:
            logger.debug(f"Could not find local tokenizer for model {llm_model}. Attempting download...")
            self.tokenizer_name = self._load_tokenizer(llm_model)
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_FOLDER / self.tokenizer_name, trust_remote_code=True)

    @override
    def num_tokens(self, messages: list["LLMMessage"]) -> int:
        conversation = [m.model_dump() for m in messages]
        try:
            tokens = self.tokenizer.apply_chat_template(conversation=conversation, add_generation_prompt=True)
        except (ValueError, TemplateError):
            text = "\n".join([message.content for message in messages])
            tokens = self.tokenizer.encode(text)
        return len(tokens)
