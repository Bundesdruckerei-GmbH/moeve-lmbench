from lmbench.metrics.wahl_o_mat import PoliticalPartiesEvaluation


def test_accuracy_requires_the_target_and_no_other_answer():
    metric = PoliticalPartiesEvaluation()
    result = metric.evaluate(
        ["Die Partei würde sagen: stimme zu.", "stimme zu, aber auch neutral", "neutral"],
        ["stimme zu", "stimme zu", "stimme zu"],
    )
    assert result["accuracy"] == [1.0, 0.0, 0.0]
    assert result["target_mentioned"] == [1.0, 1.0, 0.0]
    assert result["non_target_mentioned"] == [0.0, 1.0, 1.0]
