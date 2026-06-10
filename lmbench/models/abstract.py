"""Contains an abstract LLM class that can be used as a base class for LLMs."""

import hashlib
import logging
import threading
from abc import ABC
from typing import TYPE_CHECKING, Any

from ecologits.utils.range_value import RangeValue
from tenacity import before_sleep_log, retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

from lmbench import utils
from lmbench.config.config import CACHE_FOLDER, CONFIG
from lmbench.models.cache_manager import CacheManager
from lmbench.models.data_models import LLMMessage, LLMOutput
from lmbench.models.exceptions import ContextSizeException
from lmbench.reasoning_parser import REASONING_PARSER_MAP, ReasoningParser
from lmbench.tokenizer import TOKENIZER_MAP

if TYPE_CHECKING:
    from lmbench.tokenizer.abstract import Tokenizer

logger = logging.getLogger(__name__)


def convert_to_param_range(parameter: int | float | str) -> int | float | RangeValue:
    """Helper function to convert a model parameter into the correct format for active or total parameters.

    As for some models the exact number of parameters is not known, the sustainability evaluation has to work with
    approximations in the form of parameter ranges. If the total and active parameter count is known (e.g., 7b), then
    it can be set as an integer or float (e.g., 7.0). If it is a range, the model parameter can be given as a string
    `[min]-[max]` (e.g., 5-20). This function parses the range given in this format and transforms it into a RangeValue
    that can be read by ecologits (the library used for the sustainability evaluation).

    Args:
        parameter (int | float | str): Number of parameters in billions as an int or a float, or a range of parameters
            in billions (e.g., 5-20).

    Returns:
        int | float | RangeValue: Correct format for the use in the sustainability evaluation
    """
    if isinstance(parameter, str):
        range_split = parameter.split("-")
        assert len(range_split) == 2, "Parameter range value has to be of format [min]-[max]."
        return RangeValue(min=float(range_split[0]), max=float(range_split[1]))
    return parameter


class LLM(ABC):
    """An abstract LLM that does not implement any tasks.

    This class can be used as a base class for LLMs.
    """

    arg_string: str | None = None
    model: str
    tokenizer: "Tokenizer"
    ctx_size: int = 4096  # Default context size if nothing is set
    num_predict: int = 2048  # Default number of max tokens to generate
    cache: CacheManager
    reasoning_parser: ReasoningParser | None = None
    total_parameters: float | int | RangeValue = 0  # In billion
    active_parameters: float | int | RangeValue = 0  # In billion
    temperature: float | None = None
    cache_hit_counter: int = 0

    @staticmethod
    def name() -> str:
        """Returns the name of the LLM provider."""
        raise NotImplementedError()

    def __init__(self, model: str, **kwargs: Any):
        """Initialize abstract LLM with a model name and any named arguments and store them in the instance.

        Args:
            model (str): The name of the model that should be used.
            kwargs (dict[str, Any]): Additional keyword arguments to pass to the model.
        """
        self.model = model

        if "arg_string" in kwargs:
            self.arg_string = kwargs.pop("arg_string")

        if "ctx_size" in kwargs:
            self.ctx_size = kwargs.pop("ctx_size")

        if "num_predict" in kwargs:
            self.num_predict = kwargs.pop("num_predict")

        if "total_parameters" in kwargs:
            self.total_parameters = convert_to_param_range(kwargs.pop("total_parameters"))

        if "active_parameters" in kwargs:
            self.active_parameters = convert_to_param_range(kwargs.pop("active_parameters"))
        else:
            self.active_parameters = self.total_parameters

        if "temperature" in kwargs:
            self.temperature = kwargs.pop("temperature")

        self._kwargs: dict[str, Any] = kwargs

        if "tokenizer" in kwargs:
            tokenizer_name = kwargs.pop("tokenizer")
        else:
            tokenizer_name = "simple"

        tkn_cls = TOKENIZER_MAP[tokenizer_name]
        self.tokenizer = tkn_cls(self)

        if "reasoning_parser" in kwargs:
            reasoning_parser_name = kwargs.pop("reasoning_parser")
            self.reasoning_parser = REASONING_PARSER_MAP[reasoning_parser_name]()

        self.lock = threading.Lock()
        self.cache = CacheManager(CACHE_FOLDER / CONFIG.cache_file, self.lock, self.model)

    @classmethod
    def from_arg_string(
        cls,
        model: str,
        arg_string: str,
        **extra_kwargs: bool | str | int | float,
    ) -> "LLM":
        """
        Creates an instance of the LLM class using the given argument string and additional parameters.

        Args:
            model (str): The name of the model that should be used.
            arg_string (str): A string containing arguments in the format key1=value1,key2=value2.
            **extra_kwargs (bool | str | int | float): Additional keyword arguments forwarded to the LLM constructor.

        Returns:
            LLM: Instance of the LLM class.
        """
        params = utils.simple_parse_args_string(arg_string)
        params.update(extra_kwargs)
        return cls(model=model, arg_string=arg_string, **params)

    def hash(self) -> str:
        """Returns a hash that is unique for the general model but is stable over different runs."""
        hashobj = hashlib.sha256()
        hashobj.update(self.name().encode("utf-8"))
        hashobj.update(self.model.encode("utf-8"))
        return hashobj.hexdigest()

    def cached_invoke(self, messages: list[LLMMessage]) -> LLMOutput | str:
        """Invoke the LLM with a list of messages using a cache if available.

        Args:
            messages (list[LLMMessage]): The messages that are given to the llm. This is usually an instruction in form
                of a system message and some context as a user message.

        Returns:
            LLMOutput | str: The output of the LLM as a string or an LLMOutput object if a reasoning parser is
                available.
        """
        msg_hash = hashlib.md5(str(messages).encode("utf-8")).hexdigest()
        cached = self.cache.get(msg_hash)
        if cached is not None:
            logger.debug(f"Cache hit with message hash {msg_hash}")
            self.cache_hit_counter += 1
            return LLMOutput.from_string(cached)

        # miss → call through (with retry) and cache the full result
        result = self.invoke(messages=messages)
        # We always store the original output in the cache
        if isinstance(result, LLMOutput):
            cached_result = result.original_output
        else:
            cached_result = result
        self.cache.set(msg_hash, cached_result)
        return result

    def clear_cache(self):
        """Clear the cache for this LLM configuration."""
        self.cache.clear()

    @retry(
        before_sleep=before_sleep_log(logger, logging.INFO),
        retry=retry_if_not_exception_type(ContextSizeException),
        stop=stop_after_attempt(CONFIG.retry.stop_after_attempt),
        wait=wait_random_exponential(min=CONFIG.retry.min_wait, max=CONFIG.retry.max_wait),
    )
    def invoke(self, messages: list[LLMMessage]) -> LLMOutput | str:
        """The invoke method executes the LLM with a list of messages that can have specific roles.

        Args:
            messages (list[LLMMessage]): The messages that are given to the llm. This is usually an instruction in form
                of a system message and some context as a user message.

        Raises:
            ContextSizeException: Raised when the context size of the model is exceeded.

        Returns:
            LLMOutput | str: The output of the LLM as a string
        """
        model_output = self._invoke(messages=messages)

        if self.reasoning_parser is not None:
            model_output = self.reasoning_parser.extract_reasoning_content(model_output)

        if isinstance(model_output, LLMOutput):
            logger.debug(
                f"Reasoning output detected - {len(model_output)} ouput, "
                f"{len(model_output.reasoning_output or '')} reasoning"
            )

        return model_output

    def _invoke(self, messages: list[LLMMessage]) -> str:
        """The invoke method executes the LLM with a list of messages that can have specific roles.
        This method should be overriden.

        Args:
            messages (list[LLMMessage]): The messages that are given to the llm. This is usually an instruction in form
                of a system message and some context as a user message.

        Raises:
            ContextSizeException: Raised when the context size of the model is exceeded.

        Returns:
            str: The output of the LLM as a string
        """
        raise NotImplementedError()

    def check_tokens(self, messages: list[LLMMessage]) -> float:
        """Checks the number of tokens generate by a list of LLMMessages and returns the size relative to the
        context_window. The return value represents how much of the context window is filled by the message.

        Values less than or equal to one represent that the message fits into the context window, while messages with
        values larger than one do not fit into the context size.

        The return value can be used to scale down the message length in a way that the resulting message fits into the
        context size of the model.

        Args:
            messages (list[LLMMessage]): The messages that will be given to the llm.

        Returns:
            float: How much of the context window of the llm is filled by the message.
        """
        n_tokens = self.tokenizer.num_tokens(messages)
        return n_tokens / (self.ctx_size - self.num_predict)
