"""Defines the abstract base class for all metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod

from lmbench.dataset.abstract import Dataset
from lmbench.metrics.aggregate import AggregationType, NanPolicy
from lmbench.models.abstract import LLM
from lmbench.task import Task


class Metric(ABC):
    """Abstract base class for all metrics."""

    name: str
    tasks: list[Task]
    aggregations: list[AggregationType] | None = None
    nan_policy: NanPolicy = NanPolicy.ZERO

    @abstractmethod
    def description(self) -> str:
        """Returns a human-readable description of the metric.

        Raises:
            NotImplementedError: This method is not implemented as this class is abstract.

        Returns:
            str: A description of the metric.
        """
        raise NotImplementedError

    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, list[float]] | dict[str, float]:
        """
        Evaluate model outputs against ground-truth labels.

        Implementations must return a dictionary that maps a score name to
        either

        • a list[float] containing one value per evaluated sample, or
        • a single aggregated float that already summarises the metric.

        The two input lists must have the same length and order so that each
        prediction is compared with its corresponding label.

        Args:
            model_output : list[str]
                The model predictions.
            labels : list[str]
                The reference answers.

        Returns:
            dict[str, list[float] | float]
                Mapping from score name to per-sample scores (list) or to an
                already aggregated scalar value.
        """
        raise NotImplementedError


class LLMAwareMetric(Metric):
    """A metric that is aware of LLMs and can be used to evaluate LLM outputs."""

    def __init__(self, model: LLM) -> None:
        """Initializes the metric with an LLM for evaluation."""
        super().__init__()
        self.model = model


class DatasetAwareMetric(Metric):
    """A metric that is aware of a dataset and can use dataset-specific information for evaluation."""

    def __init__(self, dataset: Dataset, selected_indices: list[int] | None = None) -> None:
        """Initializes the metric with a dataset. The dataset may be used to evaluate with other information than the
        model output and the labels.

        Args:
            dataset (Dataset, optional): The dataset on which the metric will be evaluated.
            selected_indices (list[int] | None, optional): Dataset row indices included in the current evaluation.
        """
        super().__init__()
        self.dataset = dataset
        self.selected_indices = selected_indices

    def get_selected_column(self, name: str) -> list:
        """Return dataset column values limited to the currently evaluated row selection."""
        values = list(self.dataset.get_column(name))
        if self.selected_indices is None:
            return values
        return [values[i] for i in self.selected_indices]
