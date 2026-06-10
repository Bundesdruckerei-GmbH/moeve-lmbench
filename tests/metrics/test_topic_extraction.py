import pytest

from lmbench.metrics.topic_extraction import TopicExtraction


def test_same_input():
    sste = TopicExtraction()

    result = sste.evaluate(["keyword1,keyword2,keyword3"], ["keyword1,keyword2,keyword3"])
    assert pytest.approx(result["f1"][0], 0.001) == 1.0, "F1 score should be 1.0 for same input."
    assert pytest.approx(result["precision"][0], 0.001) == 1.0, "Precision should be 1.0 for same input."
    assert pytest.approx(result["recall"][0], 0.001) == 1.0, "Recall should be 1.0 for same input."


def test_shuffeled_input():
    sste = TopicExtraction()
    result = sste.evaluate(["keyword1,keyword2,keyword3"], ["keyword3,keyword2,keyword1"])
    assert pytest.approx(result["f1"][0], 0.001) == 1.0, "F1 score should be 1.0 for same input."
    assert pytest.approx(result["precision"][0], 0.001) == 1.0, "Precision should be 1.0 for same input."
    assert pytest.approx(result["recall"][0], 0.001) == 1.0, "Recall should be 1.0 for same input."


def test_different_input():
    sste = TopicExtraction()
    result = sste.evaluate(["keyword1,keyword2,keyword3"], ["keyword4,keyword5,keyword6"])
    assert pytest.approx(result["f1"][0], 0.001) == 0.0, "F1 score should be 0.0 for different input."
    assert pytest.approx(result["precision"][0], 0.001) == 0.0, "Precision should be 0.0 for different input."
    assert pytest.approx(result["recall"][0], 0.001) == 0.0, "Recall should be 0.0 for different input."


def test_partial_overlap():
    sste = TopicExtraction()
    result = sste.evaluate(["keyword1,keyword2,keyword3"], ["keyword1,keyword2,keyword4"])
    assert pytest.approx(result["f1"][0], 0.001) == 0.666, "F1 score should be 0.666 for partial overlap."
    assert pytest.approx(result["precision"][0], 0.001) == 0.666, "Precision should be 0.666 for partial overlap."
    assert pytest.approx(result["recall"][0], 0.001) == 0.666, "Recall should be 0.666 for partial overlap."


def test_precision():
    sste = TopicExtraction()
    result = sste.evaluate(["keyword1,keyword2,keyword3"], ["keyword1,keyword2,keyword3,keyword4"])
    assert pytest.approx(result["precision"][0], 0.001) == 1.0, "Precision should be 1.0 for all keywords in output."
    assert pytest.approx(result["recall"][0], 0.001) == 0.75, "Recall should be 0.75 for 3 out of 4 keywords in output."
    assert pytest.approx(result["f1"][0], 0.001) == 0.857, "F1 score should be 0.857 for precision and recall."


def test_recall():
    sste = TopicExtraction()
    result = sste.evaluate(["keyword1,keyword2,keyword3,keyword4"], ["keyword1,keyword2,keyword3"])
    assert pytest.approx(result["precision"][0], 0.001) == 0.75, (
        "Precision should be 0.75 for 3 out of 4 keywords in output."
    )
    assert pytest.approx(result["recall"][0], 0.001) == 1.0, "Recall should be 1.0 for all keywords in output."
    assert pytest.approx(result["f1"][0], 0.001) == 0.857, "F1 score should be 0.857 for precision and recall."
