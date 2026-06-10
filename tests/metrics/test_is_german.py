from pathlib import Path

import pytest

from lmbench.config.config import RESOURCES_FOLDER
from lmbench.metrics.is_german import IsGerman

_MODEL_PATH = Path(RESOURCES_FOLDER) / "fasttext/lid.176.bin"


@pytest.fixture
def metric():
    if not _MODEL_PATH.is_file():
        pytest.skip(f"FastText model not found at {_MODEL_PATH}")
    return IsGerman()


def test_evaluate_all_german(metric):
    model_output = [
        "Das ist ein Test. Heute ist ein schöner Tag, und wir gehen spazieren.",
        "Wie geht es Ihnen? Ich hoffe, dass alles gut ist und Sie einen angenehmen Tag haben.",
    ]
    result = metric.evaluate(model_output, [])
    expected_flags = [1, 1]
    assert result["is_german"] == expected_flags


def test_evaluate_mixed_languages(metric):
    model_output = [
        "Das ist ein Test. Heute ist ein schöner Tag, und wir gehen spazieren.",
        "This is a test. Today is a beautiful day, and we are going for a walk.",
    ]
    result = metric.evaluate(model_output, [])
    expected_flags = [1, 0]
    assert result["is_german"] == expected_flags


def test_evaluate_non_german(metric):
    model_output = [
        "This is a test. Today is a beautiful day, and we are going for a walk.",
        "Ceci est un test. Aujourd'hui est une belle journée, et nous allons nous promener.",
    ]
    result = metric.evaluate(model_output, [])
    expected_flags = [0, 0]
    assert result["is_german"] == expected_flags


def test_evaluate_short_texts(metric):
    # it is not working for short phrases, as can be seen in this test
    model_output = ["Emmanuel Macron", "1986"]
    result = metric.evaluate(model_output, [])
    expected_flags = [0, 0]
    assert result["is_german"] == expected_flags


def test_evaluate_real_data(metric):
    model_output = [
        ">BGE 103 V 46: Bedürftige Parteien mit Ansprüchen aus dem AlVG (nicht offensichtlich aussichtslos) sind im"
        " kantonalen Beschwerdeverfahren berechtigt, unentgeltliche Verbeiständung gemäss BGE 98 V 116 zu beanspruchen,"
        " da das AlVG keine spezifische Bestimmung für diese Rechtspflege enthält.",
        "The assistant notes that the assistant is already well-versed in the topic and has a thorough understanding of"
        " the relevant laws, including those related to Rechtsprechung and Rekursverfahren. The assistant therefore "
        "does not need further clarification or explanation on this matter, but rather provides an analysis based on "
        "the provided information. "
        "The assistant notes that there are three main points addressed in the Beschwerde-und Rekursverfahren Nr. "
        "1/2015: 1) Rechtsprechung für das Urteilt aus dem Betreibungsregister, zugleicht der Beschwerde-und "
        "Rekursverfährer, wenn die Beschwerdenstatistik verweigert ist. The assistant notes that the Beschwerdes- und "
        "Rekursverfahren require regular audits and statistical reports to ensure compliance with relevant laws. "
        "Failure to provide these reports can lead to penalties under Article 68 of the Swiss Civil Code (GebtschKG). "
        "2) Schutzgleichheit für einzelne Eigenturen auf dem Grund der Urteiltes, die in der Beschwerde-und "
        "Rekursverfahren erlaubt werden. The assistant explains that the purpose of these procedures is to ensure equal"
        " treatment for all parties involved. In the event of unequal treatment, penalties may be imposed as per "
        "Article 68 of the GebTSchKG (Güteutungssystemsgesetz). 3) Dritte Seite der Rekursverfahren, das betreffenden "
        "Beschwerde-und Rekursverfahren in dem Sinn behalbt die Grundsatz der Unabhängigkeit von betriebenen Beträgen. "
        "The assistant states that the third part of the Rekursverfahren is crucial for ensuring the independence of "
        "financial transactions. This involves investigating and determining whether or not a financial transaction, "
        "such as an unpaid bill, has been resolved in accordance with law.",
    ]
    result = metric.evaluate(model_output, [])
    expected_flags = [1, 0]
    assert result["is_german"] == expected_flags
