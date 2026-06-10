"""Sets the version of LMBench."""

import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)
