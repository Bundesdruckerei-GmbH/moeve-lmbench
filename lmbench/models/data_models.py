"""Data model definitions for LLM messages and roles of these messages."""

import re
import typing
from enum import Enum

from pydantic import BaseModel


class LLMRole(str, Enum):
    """Enum for the roles in a conversation.

    Attributes:
        SYSTEM (str): The system role. This is typically used to provide instructions or context to the model.
        USER (str): The user role. This is where users input their questions or prompts.
        ASSISTANT (str): The assistant role. This is where the model outputs its responses.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LLMMessage(BaseModel):
    """
    A Pydantic model representing a message in an LLM conversation.

    Attributes:
        role (str): The role of the sender, e.g., 'user' or 'assistant'.
        content (str): The content of the message.
    """

    role: LLMRole
    content: str


class LLMOutput(str):
    """A custom string class to hold the output of an LLM with additional attributes.

    Attributes:
        reasoning_output (str | None): The reasoning output of the LLM.
        original_output (str): The original output of the LLM. This is the output before any reasoning content is
           extracted or removed.
    """

    reasoning_output: str | None
    original_output: str

    def __new__(cls, result: str, reasoning_output: str | None = None, original_output: str | None = None):
        """Create a new LLMOutput instance."""
        obj = super().__new__(cls, result)
        obj.reasoning_output = reasoning_output
        if original_output is not None:
            obj.original_output = original_output
        else:
            obj.original_output = result
        return obj

    @classmethod
    def from_string(
        cls, llm_output: str, think_start_token: str = "<think>", think_end_token: str = "</think>"
    ) -> typing.Self | str:
        """Parses a string into an LLM output.

        Returns the original string if no reasoning content could be detected.

        Args:
            llm_output (str): The string to parse.
            think_start_token (str): The token that starts the reasoning content.
            think_end_token (str): The token that ends the reasoning content.

        Returns:
            LLMOutput | str: A parsed LLMOutput or the original string.
        """
        # Hot fix for cases where the LLM (provider) does not contain the opening think tag.
        if think_end_token in llm_output and think_start_token not in llm_output:
            llm_output = think_start_token + llm_output

        reasoning_regex = re.compile(rf"{think_start_token}(.*?){think_end_token}", re.DOTALL)
        matches = list(reasoning_regex.finditer(llm_output))

        if not matches:
            # no reasoning, just return the original as result
            return llm_output

        # collect all reasoning blocks (strip each for cleanliness)
        reasoning_contents = [m.group(1).strip() for m in matches]

        # remove all matched <think>…</think> spans to form the final result
        clean_parts = []
        last = 0
        for m in matches:
            start, end = m.span()
            clean_parts.append(llm_output[last:start])
            last = end
        clean_parts.append(llm_output[last:])
        clean_output = "".join(clean_parts).strip()

        return LLMOutput(
            result=clean_output, reasoning_output="\n".join(reasoning_contents), original_output=llm_output
        )
