import numpy as np
import pytest

from lmbench.metrics.aggregate import AggregationType, NanPolicy, aggregate


def test_aggregation():
    sample_scores = {
        "score1": [0.1, 0.2, 3000.0],
        "score2": [1.0, 1.0, 1.0],
        "score3": [0.25, 0.5, 0.81],
        "score4": 4.0,
    }
    agg_result = aggregate(scores=sample_scores, aggregations=[AggregationType.MEAN, AggregationType.MEDIAN])

    expected_keys = [
        "score1:mean",
        "score1:median",
        "score2:mean",
        "score2:median",
        "score3:mean",
        "score3:median",
        "score4",
    ]

    assert len(agg_result.keys()) == len(expected_keys), (
        f"Expected {len(expected_keys)} keys in the result but got {len(agg_result.keys())}"
    )
    for key in expected_keys:
        assert key in agg_result, f"Key '{key}' not found in the result"

    assert agg_result["score1:mean"] == pytest.approx(1000.1, 0.001), "Incorrect mean value for score1"
    assert agg_result["score1:median"] == pytest.approx(0.2, 0.001), "Incorrect median value for score1"
    assert agg_result["score2:mean"] == pytest.approx(1.0, 0.001), "Incorrect mean value for score2"
    assert agg_result["score2:median"] == pytest.approx(1.0, 0.001), "Incorrect median value for score2"
    assert agg_result["score3:mean"] == pytest.approx(0.52, 0.001), "Incorrect mean value for score3"
    assert agg_result["score3:median"] == pytest.approx(0.5, 0.001), "Incorrect median value for score3"
    assert agg_result["score4"] == sample_scores["score4"]


def test_nans_in_aggregation():
    sample_scores = {
        "score1": [0.1, np.nan, 3000.0],
        "score2": [np.nan, np.nan, np.nan],
        "score3": [0.25, 0.5, 0.81],
        "score4": 4.0,
    }
    agg_result = aggregate(scores=sample_scores, aggregations=[AggregationType.MEAN, AggregationType.MEDIAN])
    assert agg_result["score1:mean"] == pytest.approx(1000.0333, 0.001), "Incorrect mean value for score1"
    assert agg_result["score1:median"] == pytest.approx(0.1, 0.001), "Incorrect median value for score1"
    assert agg_result["score2:mean"] == 0.0, "Incorrect mean value for score2"


def test_nan_policy_omit_drops_nans():
    """NanPolicy.OMIT aggregates only over non-NaN entries.

    """
    scores = {
        "score": [0.4, np.nan, 0.6, np.nan],
    }
    omit_result = aggregate(
        scores=scores,
        aggregations=[AggregationType.MEAN],
        nan_policy=NanPolicy.OMIT,
    )
    zero_result = aggregate(
        scores=scores,
        aggregations=[AggregationType.MEAN],
        nan_policy=NanPolicy.ZERO,
    )

    # OMIT: mean of [0.4, 0.6] = 0.5
    assert omit_result["score:mean"] == pytest.approx(0.5)
    # ZERO: mean of [0.4, 0.0, 0.6, 0.0] = 0.25
    assert zero_result["score:mean"] == pytest.approx(0.25)
