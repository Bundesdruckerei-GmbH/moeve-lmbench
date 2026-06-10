"""Contains the exceptions raised by the llm models."""

from typing import override


class ContextSizeException(Exception):
    """Exception raised when the context size is too large for the model.

    Attributes:
        shrink_factor (float): The factor by which the context size should be shrunk.
        message (str | None): A message describing the error. Defaults to None.
    """

    shrink_factor: float
    message: str | None

    def __init__(self, shrink_factor: float, message: str | None = None):
        """Creates a context size exception with the shrink factor and message.

        Args:
            shrink_factor (float): The factory by which the message needs to shrink.
            message (str | None, optional): The original error message. Optional.
        """
        super().__init__()
        self.shrink_factor = shrink_factor
        self.message = message

    @override
    def __str__(self):
        return f"{self.shrink_factor}: {self.message}"
