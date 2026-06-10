"""This module contains the metrics that are based on scikit-learn."""

import ast
from typing import override

import pandas as pd
from sklearn.metrics import classification_report, jaccard_score

from lmbench.metrics.abstract import Metric
from lmbench.metrics.utils import parse_llm_label
from lmbench.task import Task


class ClassificationReport(Metric):
    """Standard classification metrics.

    ClassificationReport culculates the standard classification metrics macro F1, precision and recall, as well as the
    respective scores per class.

    It is based on scikit-learn's classification report.
    """

    name: str = "ClassificationReport"
    tasks: list[Task] = [Task.CLASSIFICATION]

    @override
    def description(self) -> str:
        return "None"

    @override
    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, float]:
        y_true_all = list(map(int, labels))
        num_classes = max(y_true_all) + 1  # assumes labels start at 0
        parsed = [parse_llm_label(p, num_classes=num_classes) for p in model_output]
        y_pred_all, invalid_flags = zip(*parsed)

        y_pred = [p for p, bad in zip(y_pred_all, invalid_flags) if not bad]
        y_true = [t for t, bad in zip(y_true_all, invalid_flags) if not bad]

        invalid_ratio = sum(invalid_flags) / len(invalid_flags)

        report = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True, zero_division=0)  # type: ignore[reportArgumentType]

        report = self._flatten_dict(report)  # type: ignore[reportArgumentType]
        report["invalid_ratio"] = invalid_ratio
        return report

    def _flatten_dict(self, d: dict) -> dict[str, float]:
        flattened_dict = {}
        for key, sub_d in d.items():
            if not key == "accuracy" and not key == "weighted avg":
                for metric, value in sub_d.items():
                    if not metric == "support":
                        if key == "macro avg":
                            flattened_dict[f"macro_{metric}"] = value
                        else:
                            flattened_dict[f"{key}_{metric}"] = value

        return flattened_dict


class MultiLabelSetEvaluation(Metric):
    """Multilabel Set Evaluation.

    MultiLabelSetEvaluation is used to evaluate scenarios where multiple labels but only one output per data point
    exist. Per label set, it calculates the standard classification metrics macro F1, precision and recall, as well as
    the respective scores per class. It also calculates the Jaccard score.

    It is based on scikit-learn's classification report and jaccard implementation.
    """

    name: str = "MultilabelSetEvaluation"
    tasks: list[Task] = [Task.WAHLOMAT]

    @override
    def description(self) -> str:
        return "None"

    @override
    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, float]:
        labels_df_full = pd.DataFrame([ast.literal_eval(s) for s in labels])
        num_classes = int(labels_df_full.values.max()) + 1
        parsed = [parse_llm_label(p, num_classes=num_classes) for p in model_output]
        y_pred_all, invalid_flags = zip(*parsed)

        valid_idx = [i for i, bad in enumerate(invalid_flags) if not bad]
        invalid_ratio = sum(invalid_flags) / len(invalid_flags)
        y_pred = [y_pred_all[i] for i in valid_idx]
        labels_df = labels_df_full.iloc[valid_idx].reset_index(drop=True)

        class_names = labels_df.columns

        reports: dict[str, float] = {}
        for class_ in class_names:
            y_true = labels_df[class_]

            # classification report
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)  # type: ignore[reportArgumentType]
            report = {f"{class_}_{key}": value for key, value in self._flatten_dict(report).items()}  # type: ignore[reportArgumentType]
            reports |= report

            # Jaccard Index
            index = jaccard_score(y_true, y_pred, average="macro")
            reports[f"{class_}_jaccard_index"] = index

        reports["invalid_ratio"] = invalid_ratio
        return reports

    def _flatten_dict(self, d: dict) -> dict[str, float]:
        flattened_dict = {}
        for key, sub_d in d.items():
            if not key == "accuracy" and not key == "weighted avg":
                for metric, value in sub_d.items():
                    if not metric == "support":
                        if key == "macro avg":
                            flattened_dict[f"macro_{metric}"] = value
                        else:
                            flattened_dict[f"{key}_{metric}"] = value

        return flattened_dict
