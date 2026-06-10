from unittest.mock import MagicMock

import numpy as np
import pytest

from lmbench.metrics.semscore import SemScore


def embedding_generator(input: list[str]):
    return np.array([[1, 2, 3] for _ in range(len(input))])


def test_same_embedding():
    # Create a SemScore instance
    semscore = SemScore()

    # Mock the embedding model
    semscore.embedding_model = MagicMock()

    # Define test vectors
    same_vector = [1, 0, 0]

    # Test case: Same embeddings
    semscore.embedding_model.encode.side_effect = [np.array([same_vector]), np.array([same_vector])]
    results = semscore.evaluate(["text1"], ["text1"])
    assert pytest.approx(results["cos_sim"][0], 0.001) == 1.0, "Cosine similarity should be 1.0 for identical vectors."
    assert pytest.approx(results["norm_cos_dist"][0], 0.001) == 0.0, (
        "Normalized cosine distance should be 0.0 for identical vectors."
    )


def test_orthogonal_embedding():
    # Create a SemScore instance
    semscore = SemScore()

    # Mock the embedding model
    semscore.embedding_model = MagicMock()

    same_vector = np.array([1, 0, 0])
    orthogonal_vector = np.array([0, 1, 0])

    # Test case: Orthogonal embeddings
    semscore.embedding_model.encode.side_effect = [
        np.array([same_vector]),
        np.array([orthogonal_vector]),
    ]
    results = semscore.evaluate(["text1"], ["text2"])
    assert pytest.approx(results["cos_sim"][0], 0.001) == 0.0, "Cosine similarity should be 0.0 for orthogonal vectors."
    assert pytest.approx(results["norm_cos_dist"][0], 0.001) == 0.5, (
        "Normalized cosine distance should be 0.5 for orthogonal vectors."
    )


def test_opposite_embedding():
    # Create a SemScore instance
    semscore = SemScore()

    # Mock the embedding model
    semscore.embedding_model = MagicMock()

    same_vector = np.array([1, 0, 0])
    opposite_vector = np.array([-1, 0, 0])

    # Test case: Opposite embeddings
    semscore.embedding_model.encode.side_effect = [
        np.array([same_vector]),
        np.array([opposite_vector]),
    ]
    results = semscore.evaluate(["text1"], ["text3"])
    assert pytest.approx(results["cos_sim"][0], 0.001) == -1.0, "Cosine similarity should be -1.0 for opposite vectors."
    assert pytest.approx(results["norm_cos_dist"][0], 0.001) == 1.0, (
        "Normalized cosine distance should be 1.0 for opposite vectors."
    )


def test_large_input():
    # Create a SemScore instance
    semscore = SemScore()

    # Mock the embedding model
    semscore.embedding_model = MagicMock()

    # Test case: Opposite embeddings
    semscore.embedding_model.encode = embedding_generator

    model_outputs = [f"model_output_{i}" for i in range(3000)]
    labels = [f"label_{i}" for i in range(3000)]
    results = semscore.evaluate(model_output=model_outputs, labels=labels)
    assert len(results["cos_sim"]) == 3000, "The number of results should match the number of inputs."
    assert len(results["norm_cos_dist"]) == 3000, "The number of results should match the number of inputs."
