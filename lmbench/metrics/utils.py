"""Shared helper functionality used across multiple metric implementations."""

import logging
import re
from collections import Counter
from statistics import median, pvariance
from typing import Any

logger = logging.getLogger(__name__)

_INT_PATTERN = re.compile(r"[-+]?\d+")  # first signed/unsigned int


def parse_llm_label(raw: str, num_classes: int) -> tuple[int | None, bool]:
    """
    Normalise arbitrary model output to an integer label.
    Returns (mapped_value, is_invalid).  When no integer or an out-of-range
    integer (0 … num_classes-1) is found the function returns ``None`` and sets ``is_invalid``
    to True.
    """
    raw = raw.strip()

    # 1) direct conversion – whole output is an int
    try:
        val = int(raw)
        if not (0 <= val < num_classes):
            logging.warning("Value %s outside expected range [0, %d] – skipped", val, num_classes - 1)
            return None, True
        return val, False
    except (ValueError, TypeError):
        pass

    # 2) look for the first integer anywhere in the text
    m = _INT_PATTERN.search(raw)
    if m:
        val = int(m.group(0))
        if not (0 <= val < num_classes):
            logging.warning("Value %s outside expected range [0, %d] – skipped", val, num_classes - 1)
            return None, True
        return val, False

    # fallback
    logging.warning("Unrecognised answer %r – skipped", raw)
    return None, True


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
