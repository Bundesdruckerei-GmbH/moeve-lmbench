"""This module contains the abstract definition of a ReasoningParser."""


class ReasoningParser:
    """
    Abstract reasoning parser class that should not be used directly.
    Provided and methods should be used in derived classes.

    It is used to extract reasoning content from the model output.
    """

    name: str

    def extract_reasoning_content(self, model_output: str) -> str:
        """
        Extract reasoning content from a complete model-generated string.

        Used for non-streaming responses where we have the entire model response
        available before sending to the client.

        Parameters:
        model_output: str
            The model-generated string to extract reasoning content from.

        Returns:
            LLMOutput: The model output with the reasoning content extracted.
        """
        raise NotImplementedError("AbstractReasoningParser.extract_reasoning_calls has not been implemented!")
