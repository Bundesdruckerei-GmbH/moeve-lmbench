"""This module contains the topic extraction metrics."""

from math import fsum
from typing import override

import numpy as np

from lmbench.metrics.abstract import Metric
from lmbench.metrics.semscore import SemScore, cosine_similarity
from lmbench.task import Task


def parse_topics(topics: str, delimiter: str | None = None) -> set[str]:
    """Parse topics from a string.

    Args:
        topics (str): A list of topics separated by a delimiter.
        delimiter (str, optional): The delimiter used to separate topics. If it is set to None, the delimiter will be
            determined automatically by the presence of common delimiters in the text. Defaults to None.

    Returns:
        list[str]: A list of topics.
    """
    if delimiter is None:
        delimiter = estimate_delimiter(topics)
    return {t.lower().strip() for t in topics.split(delimiter)}


def estimate_delimiter(topics_str: str) -> str:
    """Estimate the delimiter used to separate topics in a string.

    Args:
        topics_str (str): A topic string (either from a model output or a dataset)

    Returns:
        str: The estimated delimiter.
    """
    if "," in topics_str:
        return ","
    elif ";" in topics_str:
        return ";"
    elif "|" in topics_str:
        return "|"
    elif "\t" in topics_str:
        return "\t"
    elif topics_str.count("\n") > 1:
        return "\n"
    else:
        return ","  # Default value


class TopicExtraction(Metric):
    """Topic Extraction metric based on lowercase comparison of extracted topic with goldstandard topics.

    The following scores are returned:
     - precision: The precision of detected topics.
     - recall: The recall of detected topics.
     - f1: The F1 score of precision and recall.
    """

    name = "TopicExtraction"
    tasks = [Task.TOPIC_EXTRACTION]

    @override
    def description(self) -> str:
        return "Topic Extraction metric."

    @override
    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, list[float]]:
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        for output, label in zip(model_output, labels):
            predictions = parse_topics(output)
            ground_truth = parse_topics(label)

            precision = 0.0
            recall = 0.0
            true_positives = len(ground_truth.intersection(predictions))
            false_positives = len(predictions.difference(ground_truth))
            false_negatives = len(ground_truth.difference(predictions))
            # Calculate precision, recall, and F1 score
            if true_positives + false_positives == 0.0:
                precision = 0.0
            else:
                precision = true_positives / (true_positives + false_positives)

            if true_positives + false_negatives == 0.0:
                recall = 0.0
            else:
                recall = true_positives / (true_positives + false_negatives)
            if precision + recall == 0.0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        return {"f1": f1s, "precision": precisions, "recall": recalls}


class SemScoreTopicExtraction(Metric):
    """Topic Extraction metric based on SemScore comparison of extracted topic with goldstandard topics.

    The following scores are returned:
     - precision: The average precision of all topics as calculated by SemScore.
     - recall: The average recall of all topics as calculated by SemScore.
     - f1: The average F1 score of all topics as calculated by SemScore.
    """

    name = "SemScoreTopicExtraction"
    tasks = [Task.TOPIC_EXTRACTION]

    @override
    def __init__(self) -> None:
        super().__init__()
        self.semscore = SemScore()

    @override
    def description(self) -> str:
        return "Topic Extraction metric using SemScore for a fuzzy topic matching."

    def _get_most_similar_sem(self, candidate_embed: list[float], ground_truth_embeds: list[list[float]]) -> float:
        """Get the most similar semantic score between a candidate and a list of ground truth topics in the form of
        embedding vectors.

        Args:
            candidate_embed (list[float]): The embedding of the candidate topic.
            ground_truth_embeds (list[list[float]]): The list of embeddings of the ground truth topics.

        Returns:
            float: The most similar semantic score between the candidate and the ground truth topics.
                The score is in the range [0, 1].
        """
        return max(
            [
                (cosine_similarity(np.array(candidate_embed), np.array(ground_truth_embed)) + 1) / 2
                for ground_truth_embed in ground_truth_embeds
            ]
        )

    @override
    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, list[float]]:
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        for output, label in zip(model_output, labels):
            predictions = parse_topics(output)
            ground_truth = parse_topics(label)

            pred_embeds = self.semscore.get_embedding(list(predictions))
            gt_embeds = self.semscore.get_embedding(list(ground_truth))

            precision = fsum([self._get_most_similar_sem(candidate, gt_embeds) for candidate in pred_embeds]) / len(
                predictions
            )
            recall = fsum([self._get_most_similar_sem(gt_token, pred_embeds) for gt_token in gt_embeds]) / len(
                ground_truth
            )
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        return {"f1": f1s, "precision": precisions, "recall": recalls}
