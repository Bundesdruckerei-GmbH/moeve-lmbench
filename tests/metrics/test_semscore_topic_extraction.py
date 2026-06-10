from unittest.mock import MagicMock

import numpy as np
import pytest

from lmbench.metrics.topic_extraction import SemScoreTopicExtraction


def embedding_generator(input: list[str]):
    return np.array([[1, 2, 3] for _ in range(len(input))])


def test_same_embedding():
    sste = SemScoreTopicExtraction()

    # Mock the embedding model
    sste.semscore.embedding_model = MagicMock()

    # Define test vectors
    same_vector = [1, 0, 0]

    # Test case: Same embeddings
    sste.semscore.embedding_model.encode.side_effect = [np.array([same_vector]), np.array([same_vector])]
    results = sste.evaluate(["keyword1"], ["keyword1"])
    assert pytest.approx(results["f1"][0], 0.001) == 1.0, "F1 score should be 1.0 for identical vectors."
    assert pytest.approx(results["precision"][0], 0.001) == 1.0, "Precision should be 1.0 for identical vectors."
    assert pytest.approx(results["recall"][0], 0.001) == 1.0, "Recall should be 1.0 for identical vectors."


def test_orthogonal_embedding():
    sste = SemScoreTopicExtraction()

    # Mock the embedding model
    sste.semscore.embedding_model = MagicMock()

    same_vector = np.array([1, 0, 0])
    orthogonal_vector = np.array([0, 1, 0])

    # Test case: Orthogonal embeddings
    sste.semscore.embedding_model.encode.side_effect = [
        np.array([same_vector]),
        np.array([orthogonal_vector]),
    ]
    results = sste.evaluate(["text1"], ["text2"])
    assert pytest.approx(results["f1"][0], 0.001) == 0.5, "F1 score should be 0.5 for orthogonal vectors."
    assert pytest.approx(results["precision"][0], 0.001) == 0.5, "Precision should be 0.5 for orthogonal vectors."
    assert pytest.approx(results["recall"][0], 0.001) == 0.5, "Recall should be 0.5 for orthogonal vectors."


def test_opposite_embedding():
    sste = SemScoreTopicExtraction()

    # Mock the embedding model
    sste.semscore.embedding_model = MagicMock()

    same_vector = np.array([1, 0, 0])
    opposite_vector = np.array([-1, 0, 0])

    # Test case: Opposite embeddings
    sste.semscore.embedding_model.encode.side_effect = [
        np.array([same_vector]),
        np.array([opposite_vector]),
    ]
    results = sste.evaluate(["text1"], ["text3"])
    assert pytest.approx(results["f1"][0], 0.001) == 0.0, "F1 score should be 0.0 for opposite vectors."
    assert pytest.approx(results["precision"][0], 0.001) == 0.0, "Precision should be 0.0 for opposite vectors."
    assert pytest.approx(results["recall"][0], 0.001) == 0.0, "Recall should be 0.0 for opposite vectors."
