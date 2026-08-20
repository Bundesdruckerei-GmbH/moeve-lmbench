"""Helpers exclusive to the Hallucination metric.

Includes the dataset's structured-response parser, the question-type
normalisation map, and score-aggregation primitives used by the multi-shot
sub-evaluators. Anything that turns out to have a second consumer should be
promoted back to ``lmbench.metrics.utils``.
"""

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from statistics import median, pvariance
from typing import Any

_HALLUCINATION_QUESTION_TYPE_ALIASES = {
    "SIMPLE": "ANSWERABLE",
    "COMPLEX": "ANSWERABLE",
}


def clamp01(value: Any, default: float = 0.0) -> float:
    """Convert a value to float and clamp it into the unit interval."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def f1_score(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall."""
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def majority_bool(values: list[bool]) -> bool:
    """Return the most common boolean value."""
    return Counter(values).most_common(1)[0][0]


def median_score(scores: list[float]) -> float:
    """Return the median score as a float."""
    return float(median(scores)) if scores else 0.0


def score_variance(scores: list[float]) -> float:
    """Return the population variance for repeated scores."""
    return float(pvariance(scores)) if len(scores) > 1 else 0.0


def closest_index(scores: list[float], target: float) -> int:
    """Return the index of the score closest to target."""
    return min(range(len(scores)), key=lambda idx: abs(scores[idx] - target))


def normalize_hallucination_question_type(question_type: Any) -> str:
    """Return the canonical hallucination question type used for scoring."""
    normalized = _as_text(question_type).upper().strip()
    return _HALLUCINATION_QUESTION_TYPE_ALIASES.get(normalized, normalized)


class ResponseStatus(Enum):
    """Canonical status labels emitted by the hallucination dataset's structured output parser."""

    ANSWERABLE = "ANSWERABLE"
    ANSWERED = "ANSWERABLE"
    BEANTWORTET = "ANSWERABLE"
    UNANSWERABLE = "UNANSWERABLE"
    UNBEANTWORTBAR = "UNANSWERABLE"
    CONFLICTING = "CONFLICTING"
    CONFLICT = "CONFLICTING"
    WIDERSPRUCH = "CONFLICTING"
    PARSE_ERROR = "PARSE_ERROR"


@dataclass
class ParsedResponse:
    """Structured representation of a parsed hallucination benchmark LLM response."""

    status: ResponseStatus
    documents: list[int]
    response_text: str
    raw_output: str


def parse_hallucination_output(raw_output: str) -> ParsedResponse:
    """Parse structured LLM response from benchmark."""
    patterns = {
        "status": (
            r"STATUS:\s*("
            r"BEANTWORTET|UNBEANTWORTBAR|WIDERSPRUCH|ANSWERED|ANSWERABLE|UNANSWERABLE|CONFLICTING|CONFLICT"
            r")"
        ),
        "documents": r"DOKUMENTE:\s*([^\n]+)",
        "response": r"ANTWORT:\s*(.*)",
    }

    status_match = re.search(patterns["status"], raw_output, re.IGNORECASE)
    docs_match = re.search(patterns["documents"], raw_output, re.IGNORECASE)
    response_match = re.search(patterns["response"], raw_output, re.IGNORECASE | re.DOTALL)

    if not status_match:
        return ParsedResponse(status=ResponseStatus.PARSE_ERROR, documents=[], response_text="", raw_output=raw_output)

    # Parse status
    status = ResponseStatus[status_match.group(1).upper()]

    # Parse document references
    documents = []
    if docs_match:
        doc_str = docs_match.group(1).strip()
        if doc_str.upper() != "NONE":
            documents = [int(d.strip()) for d in re.findall(r"\d+", doc_str)]

    # Parse response text
    response_text = response_match.group(1).strip() if response_match else ""

    return ParsedResponse(status=status, documents=documents, response_text=response_text, raw_output=raw_output)


def _as_text(value) -> str:
    if value is None:
        return ""
    return str(value)
