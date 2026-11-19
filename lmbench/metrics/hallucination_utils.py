"""Helpers exclusive to the Hallucination metric.

Includes the dataset's structured-response parser, the question-type
normalisation map. Anything that turns out to have a second consumer should be
promoted to ``lmbench.metrics.utils`` (the score-aggregation primitives already were).
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

_HALLUCINATION_QUESTION_TYPE_ALIASES = {
    "SIMPLE": "ANSWERABLE",
    "COMPLEX": "ANSWERABLE",
}


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
