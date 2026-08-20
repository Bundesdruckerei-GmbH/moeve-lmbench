from lmbench.metrics.hallucination_utils import (
    ResponseStatus,
    clamp01,
    closest_index,
    f1_score,
    majority_bool,
    median_score,
    normalize_hallucination_question_type,
    parse_hallucination_output,
    score_variance,
)


def test_clamp01_clamps_out_of_range():
    assert clamp01(-0.5) == 0.0
    assert clamp01(1.5) == 1.0
    assert clamp01(0.3) == 0.3


def test_clamp01_returns_default_for_non_numeric():
    assert clamp01("not a number") == 0.0
    assert clamp01(None, default=0.5) == 0.5


def test_f1_score_handles_zero_denominator():
    assert f1_score(0.0, 0.0) == 0.0
    assert f1_score(1.0, 1.0) == 1.0
    assert f1_score(0.5, 1.0) == 2 * 0.5 * 1.0 / 1.5


def test_majority_bool_returns_most_common():
    assert majority_bool([True, True, False]) is True
    assert majority_bool([False, False, True]) is False


def test_median_score_handles_empty():
    assert median_score([]) == 0.0
    assert median_score([1.0, 3.0, 2.0]) == 2.0
    assert median_score([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_score_variance_handles_short_inputs():
    assert score_variance([]) == 0.0
    assert score_variance([0.7]) == 0.0
    assert score_variance([1.0, 1.0, 1.0]) == 0.0
    # pvariance of [0, 1] = 0.25
    assert score_variance([0.0, 1.0]) == 0.25


def test_closest_index_picks_nearest():
    assert closest_index([0.1, 0.5, 0.9], 0.6) == 1
    # ties: min() returns the first occurrence
    assert closest_index([0.0, 1.0], 0.5) == 0


def test_normalize_question_type_maps_dataset_aliases():
    assert normalize_hallucination_question_type("SIMPLE") == "ANSWERABLE"
    assert normalize_hallucination_question_type("complex") == "ANSWERABLE"
    assert normalize_hallucination_question_type(" UNANSWERABLE ") == "UNANSWERABLE"
    # unknown values pass through normalised
    assert normalize_hallucination_question_type("weird") == "WEIRD"
    assert normalize_hallucination_question_type(None) == ""


def test_parse_hallucination_output_german_status_aliases():
    raw = "STATUS: BEANTWORTET\nDOKUMENTE: 1, 3\nANTWORT: Berlin ist die Hauptstadt."
    parsed = parse_hallucination_output(raw)
    assert parsed.status is ResponseStatus.BEANTWORTET
    assert parsed.status.value == "ANSWERABLE"
    assert parsed.documents == [1, 3]
    assert parsed.response_text == "Berlin ist die Hauptstadt."


def test_parse_hallucination_output_handles_none_documents():
    raw = "STATUS: UNBEANTWORTBAR\nDOKUMENTE: NONE\nANTWORT: Keine Information."
    parsed = parse_hallucination_output(raw)
    assert parsed.status.value == "UNANSWERABLE"
    assert parsed.documents == []
    assert parsed.response_text == "Keine Information."


def test_parse_hallucination_output_conflict_status():
    raw = "STATUS: WIDERSPRUCH\nDOKUMENTE: 2\nANTWORT: Quellen widersprechen sich."
    parsed = parse_hallucination_output(raw)
    assert parsed.status.value == "CONFLICTING"
    assert parsed.documents == [2]


def test_parse_hallucination_output_missing_status_returns_parse_error():
    parsed = parse_hallucination_output("ANTWORT: Nur Text, kein Status.")
    assert parsed.status is ResponseStatus.PARSE_ERROR
    assert parsed.documents == []
    assert parsed.response_text == ""
    assert parsed.raw_output.startswith("ANTWORT:")


def test_parse_hallucination_output_multiline_answer():
    raw = "STATUS: ANSWERABLE\nDOKUMENTE: 1\nANTWORT: Zeile eins.\nZeile zwei."
    parsed = parse_hallucination_output(raw)
    assert parsed.response_text == "Zeile eins.\nZeile zwei."
