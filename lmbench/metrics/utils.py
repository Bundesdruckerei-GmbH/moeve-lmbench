"""Shared helper functionality used across multiple metric implementations."""

import logging
import re

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
