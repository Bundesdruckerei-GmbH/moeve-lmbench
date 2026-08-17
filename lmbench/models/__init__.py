"""This module contains the auxiliary functions to load and save the LLM models. Whenever there is
a new LLM to be tested, it should be added to the TYPE_MAPPING dictionary in this module. The key
needs to be the type of the LLM as a string (as defined in the MetaData object), and the value needs
to be the class of the LLM.

Additionally, initial configurations can be added to the generate_model_list() function.
"""

from lmbench.models.abstract import LLM
from lmbench.models.anthropic import AnthropicLLM
from lmbench.models.ollama import OllamaLLM
from lmbench.models.openai import AzureOpenAILLM, OpenAILLM

MODEL_LIST: list[type[LLM]] = [OllamaLLM, OpenAILLM, AzureOpenAILLM, AnthropicLLM]
MODEL_MAPPING: dict[str, type[LLM]] = {m.name(): m for m in MODEL_LIST}
