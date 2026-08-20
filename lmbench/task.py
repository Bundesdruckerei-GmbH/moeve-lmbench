"""Contains the task class that defined which tasks can be run by LMBench."""

from __future__ import annotations

import enum
import logging

logger = logging.getLogger(__name__)


class Task(str, enum.Enum):
    """Task enum for the task to be performed by the model."""

    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    CLASSIFICATION = "classification"
    TOPIC_EXTRACTION = "topic_extraction"
    WAHLOMAT = "wahl_o_mat"
    SEMANTICSIMILARITY = "semantic_similarity"
    POLITICAL_PARTIES = "political_parties"
    VALUE_EVALUATION = "value_evaluation"
    HALLUCINATION = "hallucination"
