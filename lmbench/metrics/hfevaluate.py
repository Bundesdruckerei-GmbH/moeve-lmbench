"""This module contains the metrics that are based on the huggingface/evaluate library."""

from typing import override

import evaluate
import torch

from lmbench.metrics.abstract import Metric
from lmbench.task import Task


class ROUGE(Metric):
    """ROUGE metric.

    ROUGE is a metric for evaluating text summarization and machine translation. It is based on
    n-gram co-occurence statistics.

    This ROUGE implementation is based on the huggingface/evaluate library and returns the following

    scores:
     - rouge1: ROUGE-1, the overlap of 1-grams between the prediction and reference.
     - rouge2: ROUGE-2, the overlap of bigrams between the prediction and reference.
     - rougeL: Longest Common Subsequence (LCS), the longest sequence of words that are the same
     - rougeLsum: ROUGE-Lsum, the sum of LCS divided by the total number of words in reference
    """

    @override
    def __init__(self) -> None:
        super().__init__()
        self.rouge = evaluate.load("rouge")

    name: str = "ROUGE"
    tasks: list[Task] = [Task.SUMMARIZATION, Task.QUESTION_ANSWERING, Task.SEMANTICSIMILARITY]

    @override
    def description(self) -> str:
        return self.rouge.description

    @override
    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, float]:
        result = self.rouge.compute(predictions=model_output, references=labels)
        if result is None:
            return {}
        return {key: float(value) for key, value in result.items()}


class BLEU(Metric):
    """BLEU metric.

    BLEU is a metric for evaluating text summarization and machine translation. It is based on the
    n-gram co-occurence statistics and is inspired by the precision metric.

    This BLEU implementation is based on the huggingface/evaluate library and returns the following
    scores:
     - bleu: BLEU score.
     - precision: The precision score which is the average of the precision of all input sentences.
    """

    @override
    def __init__(self) -> None:
        super().__init__()
        self.bleu = evaluate.load("bleu")

    name: str = "BLEU"
    tasks: list[Task] = [Task.SUMMARIZATION, Task.QUESTION_ANSWERING, Task.SEMANTICSIMILARITY]

    @override
    def description(self) -> str:
        return self.bleu.description

    @override
    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, float]:
        result = self.bleu.compute(predictions=model_output, references=labels)
        if result is None:
            return {}
        return {
            "bleu": result["bleu"],
        }


class BERTScore(Metric):
    """BERTScore metric.

    BERTScore is a metric for evaluating text summarization and machine translation. It is based on
    the BERT language model.

    This BERTScore implementation is based on the huggingface/evaluate library and returns the
    following scores:
     - precision: The precision score which is the average of the precision of all input sentences.
     - recall: The recall score which is the average of the recall of all input sentences.
     - f1: The F1 score which is the average of the F1 of all input sentences.
    """

    def __init__(self, lang: str = "de") -> None:
        """Initializes the BERTScore metric with the given language.

        Args:
            dataset(Dataset, optional): The dataset to evaluate on. Not needed for this metric.
            lang (str, optional): The language code of the data to evaluate. Defaults to "de".
        """
        super().__init__()
        self.bertscore = evaluate.load("bertscore")
        self.lang = lang

    name: str = "BERTScore"
    tasks: list[Task] = [Task.SUMMARIZATION, Task.QUESTION_ANSWERING, Task.SEMANTICSIMILARITY]

    @override
    def description(self) -> str:
        return self.bertscore.description

    @override
    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, list[float]]:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        result = self.bertscore.compute(
            predictions=model_output, references=labels, lang=self.lang, batch_size=1, device=device
        )
        if result is None:
            return {}
        return {
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
        }
