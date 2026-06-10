"""This module contains custom metrics for individual datasets."""

import ast
from typing import override

import pandas as pd

from lmbench.metrics.abstract import Metric
from lmbench.metrics.utils import parse_llm_label
from lmbench.task import Task


class PoliticalPartiesEvaluation(Metric):
    """Political Parties Custom Metric."""

    name: str = "Political-Parties"
    tasks: list[Task] = [Task.POLITICAL_PARTIES]

    @override
    def description(self) -> str:
        return "Evaluates the model's ability to classify political party positions."

    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, list[float]]:
        """
        Evaluate model outputs against ground-truth labels for political party classification.

        Args:
            model_output : list[str]
                The model predictions in the original order of theses.
            labels : list[str]
                Ground-truth answers, one serialised Python list per party.

        Returns:
            dict[str, list[float]]
                Accuracy of the model's predictions.
        """
        scores = []
        non_target_mentioned = []
        target_mentioned = []
        for output, label in zip(model_output, labels):
            output = output.lower()
            label = label.lower()
            possibilities = ["stimme zu", "stimme nicht zu", "neutral"]

            non_target = possibilities.copy()
            non_target.remove(label)

            scores.append(1.0 if label in output and all(nt not in output for nt in non_target) else 0.0)
            # Count non-targets in output for potential further analysis
            non_target_mentioned.append(float(sum(nt in output for nt in non_target)))
            target_mentioned.append(1.0 if label in output else 0.0)
        return {"accuracy": scores, "target_mentioned": target_mentioned, "non_target_mentioned": non_target_mentioned}


class WahlOMatEvaluation(Metric):
    """Wahl-O-Mat Custom Metric.

    WahlOMatEvaluation calculates the Wahl-O-Mat score. It follows the official `scoring matrix <https://www.bpb.de/system/files/dokument_pdf/Rechenmodell_des_Wahl-O-Mat.pdf>`_.
    """

    name: str = "Wahl-O-Mat"
    tasks: list[Task] = [Task.WAHLOMAT]

    @override
    def description(self) -> str:
        return "None"

    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, float]:
        """
        Calculate Wahl-O-Mat scores for each party/column and report the share
        of unrecognised answers.

        Args:
            model_output : list[str]
                The model’s predictions in the original order of theses.
            labels : list[str]
                Ground-truth answers, one serialised Python list per party.

        Returns:
            dict[str, float]
                Party scores **plus** an additional entry ``"invalid_ratio"``
                representing the fraction of model outputs that could not be
                mapped unambiguously and were therefore treated as neutral.
        """
        parsed = [parse_llm_label(p, num_classes=3) for p in model_output]  # labels 0,1,2
        y_pred_all, invalid_flags = zip(*parsed)
        invalid_ratio = sum(invalid_flags) / len(invalid_flags)

        # keep only rows with valid predictions
        valid_idx = [i for i, bad in enumerate(invalid_flags) if not bad]
        y_pred = [y_pred_all[i] for i in valid_idx]

        labels_df_full = pd.DataFrame([ast.literal_eval(label) for label in labels])
        labels_df = labels_df_full.iloc[valid_idx].reset_index(drop=True)

        # compute party scores
        scores: dict[str, float] = {
            f"{col}_score": self._calculate_wahlomat_score(list(y_pred), labels_df[col].tolist())
            for col in labels_df.columns
        }

        # add extra metric
        scores["invalid_ratio"] = invalid_ratio
        return scores

    def _calculate_wahlomat_score(self, y_pred: list[int], y_true: list[int]) -> float:
        agreement_scores = []
        for pred, true in zip(y_pred, y_true):
            if pred == true:  # same label
                agreement_scores.append(2)
            elif (pred == 0 and true == 2) or (pred == 2 and true == 0):  # one positive - one neutral
                agreement_scores.append(1)
            elif (pred == 1 and true == 2) or (pred == 2 and true == 1):  # one negative - one neutral
                agreement_scores.append(1)
            else:  # one positive - one negative
                agreement_scores.append(0)

        return self._min_max_norm(sum(agreement_scores), min_t=0, max_t=2 * len(agreement_scores))

    def _min_max_norm(self, score: int, min_t: int, max_t: int) -> float:
        return (score - min_t) / (max_t - min_t)
