"""This module contains the classes to aggregate metrics."""

from collections.abc import Callable
from enum import Enum

import numpy as np


class AggregationType(str, Enum):
    """The valid types of aggregations.

    Attibutes:
        MEAN: The mean aggregation.
        MEDIAN: The median aggregation
    """

    MEAN = "mean"
    MEDIAN = "median"


class NanPolicy(str, Enum):
    """How NaN values should be handled before aggregation.

    Attributes:
        ZERO: Replace NaN values with 0.0 before aggregating.
        OMIT: Drop NaN values from the list before aggregating.
    """

    ZERO = "zero"
    OMIT = "omit"


aggregation_map: dict[AggregationType, Callable[[list[float]], float]] = {
    AggregationType.MEAN: np.mean,
    AggregationType.MEDIAN: np.median,
}


def aggregate(
    scores: dict[str, list[float] | float],
    aggregations: list[AggregationType] | None = None,
    nan_policy: NanPolicy = NanPolicy.ZERO,
) -> dict[str, float]:
    """Aggregate the scores of a metric using 1 or more aggregation types given by the agg_ypes argument.

    If a score is a single float value it will be returned as is. Otherwise, if it's a list of floats, it will be
    aggreagated according to the specified aggregation types.

    When a score is aggregated, the name of the aggregation is added to the score name with a colon (:) separator.
    E.g., "precision:mean", "precision:median"

    Args:
        scores (dict[str, list[float] | float]): The result of a metric computation in the form of a dictionary
            mapping the names of the scores to either a vaule or a list of values.
        aggregations (list[AggregationType] | None, optional): The aggregations that should be applied to all scores.
            If None, all aggregations available will be applied. Defaults to None.
        nan_policy (NanPolicy): How to handle NaN values in the score lists before aggregating. ZERO replaces NaNs
            with 0.0; OMIT drops them entirely. Defaults to ZERO.

    Returns:
        dict[str, float]: A list mapping the names of the scores to their aggregated values.
    """
    if aggregations is None:
        aggregations = list(AggregationType)

    # Initialize the result dictionary
    result = {}

    for key in scores.keys():
        value = scores[key]
        if isinstance(value, list):
            if nan_policy == NanPolicy.OMIT:
                clean: list[float] = [v for v in value if not (isinstance(v, float) and np.isnan(v))]
            else:
                # Set NaNs to zero to avoid NaNs propagating through the aggregation functions.
                clean = [0.0 if isinstance(v, float) and np.isnan(v) else v for v in value]
            # Aggregate each value using the specified aggregation types
            for aggregation in aggregations:
                result[f"{key}:{aggregation.value}"] = aggregation_map[aggregation](clean)
        else:
            # If it's a single value, just copy it over
            result[key] = value

    return result
