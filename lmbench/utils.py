"""Util functions used for the benchmark."""

import logging
import os
import re
from string import Formatter
from typing import Any

from lmbench.models.exceptions import ContextSizeException

logger = logging.getLogger(__name__)


def is_gpu_environment() -> bool:
    """Checks whether the run is inside a GPU environment, which means that the LLM is executed locally.

    The check is performed using environment variables. If any of the following environment variables is set,
    it returns True:
     - USE_OLLAMA
     - USE_VLM
     - USE_LLAMACPP

    Returns:
        bool: True if the run is inside a GPU environment, False otherwise.
    """
    # Check whether ENV vars USE_OLLAMA, USE_VLM or USE_LLAMACPP are set
    use_ollama = os.environ.get("USE_OLLAMA", None) is not None
    use_vlm = os.environ.get("USE_VLM", None) is not None
    use_llamacpp = os.environ.get("USE_LLAMACPP", None) is not None
    return use_ollama or use_vlm or use_llamacpp


def simple_parse_args_string(args_string: str) -> dict[str, bool | str | int | float]:
    """
    Parse something like `args1=val1,arg2=val2` into a dictionary.

    Args:
        args_string (str): The string to parse.

    Returns:
        dict[str, bool | str | int | float]: A dictionary with the parsed arguments.
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {k: handle_arg_string(v) for k, v in [arg.split("=") for arg in arg_list]}
    return args_dict


def handle_arg_string(arg: str) -> bool | str | int | float:
    """Handle a single value string and convert it to the appropriate data type.

    Args:
        arg (str): A single value.

    Returns:
        bool | str | int | float: The data but in the deduced type.
    """
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg


def get_keys_from_format_string(format_string: str) -> list[str]:
    """Get the keys from a format string.

    Args:
        format_string (str): The format string to parse.

    Returns:
        list[str]: A list of keys.
    """
    return [i[1] for i in Formatter().parse(format_string) if i[1] is not None]


def quantization_from_model(provider: str, model_name: str) -> str | None:
    """Get the quantization from a model name.

    Args:
        provider (str): The provider of the model.
        model_name (str): The model name to parse.

    Returns:
        str | None: The quantization of the model. I.e., "4-bit", "8-bit", etc. or None if not found.
    """
    model_name = model_name.lower()
    if "q2" in model_name:
        return "2-bit"
    elif "q3" in model_name:
        return "3-bit"
    elif "q4" in model_name:
        return "4-bit"
    elif "q5" in model_name:
        return "5-bit"
    elif "q6" in model_name:
        return "6-bit"
    elif "q8" in model_name:
        return "8-bit"
    elif "fp16" in model_name or "f16" in model_name:
        return "FP16"
    elif provider == "ollama" and ":" not in model_name:
        return "4-bit"  # Per default we assume 4 bit if ollama is used
    return None


def context_size_eception_from_errormsg(errormsg: str) -> ContextSizeException:
    """Parse the context size exception from an error message. For this, the numbers contained in the error message are
    extracted and the shrink_factor is calculated as the ratio of the smaller number to the larger number.
    The exception is then created with the shrink_factor and the original error message.

    Args:
        errormsg (str): The error message raised due to exceeding context size.

    Returns:
        ContextSizeException: The exception with the calculated shrink factor and original error message.
    """
    errormsg = re.sub(r"[\(\[].*?[\)\]]", "", errormsg)  # Remove text in parentheses.
    numbers = re.findall(r"\d+", errormsg)
    numbers = [int(n) for n in numbers if n != "400"]
    assert len(numbers) == 2, f"Unexpected number of numbers ({numbers}) in error message: {errormsg}"
    smaller = min(numbers)
    larger = max(numbers)
    shrink_factor = smaller / larger
    logger.error(
        f"Context size was exceeded (tokens={larger}, ctx_size={smaller}). Now shrinking by {shrink_factor:.5f}"
    )
    return ContextSizeException(shrink_factor=shrink_factor, message=errormsg)


def map_list_to_dict(list_of_dicts: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """
    This function takes a list of dictionaries and maps it to a single dictionary.
    Each key in the resulting dictionary maps to a list of values from the input dictionaries.
    If a key is present in multiple input dictionaries, all corresponding values are included in the list.

    Args:
        list_of_dicts (List[Dict[str, Any]]): A list of dictionaries to be mapped.

    Returns:
        Dict[str, List[Any]]: A dictionary where each key maps to a list of values from the input dictionaries.
    """
    result_dict = {}
    for dict in list_of_dicts:
        for key, value in dict.items():
            if key not in result_dict:
                result_dict[key] = [value]
            else:
                result_dict[key].append(value)
    return result_dict


def sanitize_name(s: str) -> str:
    """Sanitize a string to contain only alphanumeric characters and dashes.

    Removes invalid characters and replaces spaces with dashes.

    Args:
        s: The string to sanitize.

    Returns:
        The sanitized string.
    """
    s = re.sub("[^a-zA-Z0-9]", " ", s)
    s = re.sub(" +", "-", s.strip())
    return s
