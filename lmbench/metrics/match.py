"""This module contains simple matching metrics."""

from typing import override

from lmbench.metrics.abstract import Metric
from lmbench.task import Task


class Match(Metric):
    """Match metric.

    The match metric includes ExactMatch and case insensitive and number+symbol-insensitive metrics. These are used for
    evaluating question answering systems, but can also be used for other tasks.

    The 3 metrics are:
     - exact_match: Only if the two strings match exactly.
     - case_insensitive_match: If the two strings match ignoring case.
     - fuzzy_match: If the two strings match ignoring case, numbers and symbols
    """

    name: str = "ExactMatch"
    tasks: list[Task] = [Task.SUMMARIZATION, Task.QUESTION_ANSWERING, Task.TOPIC_EXTRACTION, Task.POLITICAL_PARTIES]

    @override
    def description(self) -> str:
        return "This metric evaluates the exact match, case-insensitive match and a fuzzy match between two strings."

    def _fuzzy_compare(self, s1: str, s2: str) -> bool:
        """Compare two strings ignoring case, numbers and symbols."""
        import re

        s1 = re.sub(r"[^a-zA-Z]", "", s1).lower()
        s2 = re.sub(r"[^a-zA-Z]", "", s2).lower()

        return s1 == s2

    @override
    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, list[float]]:
        exact_match = [1.0 if a == b else 0.0 for a, b in zip(model_output, labels)]
        case_insensitive_match = [1.0 if a.lower() == b.lower() else 0.0 for a, b in zip(model_output, labels)]
        fuzzy_match = [1.0 if self._fuzzy_compare(a, b) else 0.0 for a, b in zip(model_output, labels)]
        return {
            "exact_match": exact_match,
            "case_insensitive_match": case_insensitive_match,
            "fuzzy_match": fuzzy_match,
        }
