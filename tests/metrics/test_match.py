import pytest

from lmbench.metrics.match import Match

model_output = [
    "Das ist der erste etwas längere aber nicht ganz richtig Satz.",
    "Das ist der zweite Satz, die dafür viel meer Informationen enthält und somit sich "
    "richtig gut für keine Evaluierung eignet.",
    "Ein dritter, vollständig korrekter Satz rundet den ganzen Testdatensatz ab.",
]
labels = [
    "Das ist der erste etwas längere aber nicht ganz richtige Satz.",
    "Das ist der zweite Satz, der dafür viel mehr Informationen enthält und somit sich "
    "richtig gut für eine Evaluierung eignet.",
    "Ein dritter, vollständig korrekter Satz rundet den ganzen Testdatensatz ab.",
]


def test_exact_match():
    em = Match()
    results = em.evaluate(model_output, labels)
    assert len(results) == 3, "Exact Match should return one result."
    assert "exact_match" in results, "Exact Match should return exact_match."
    assert "case_insensitive_match" in results, "Exact Match should return case_insensitive_match."
    assert "fuzzy_match" in results, "Exact Match should return fuzzy_match."

    # Exact Match
    results = em.evaluate(["Das ist ein Test"], ["Das ist ein Test"])
    assert pytest.approx(results["exact_match"][0], 0.001) == 1.0, "Exact Match should return 1.0 for a perfect match."

    results = em.evaluate(["Das ist ein Test"], ["Das ist ein Test2"])
    assert pytest.approx(results["exact_match"][0], 0.001) == 0.0, (
        "Exact Match should return 0.0 for non-matching strings."
    )

    # Case insensitive match
    results = em.evaluate(["das ist ein test"], ["Das IST ein Test"])
    assert pytest.approx(results["case_insensitive_match"][0], 0.001) == 1.0, (
        "Case insensitive Match should return 1.0 for a perfect match"
    )
    # Fuzzy match
    results = em.evaluate(
        ["D8as i$%$st e00in-12 t34est", "D8as i$%$st e00in-12 t34esto"], ["dasisteintest", "dasisteintest"]
    )
    assert pytest.approx(results["fuzzy_match"][0], 0.001) == 1.0, "Fuzzy Match should return 1.0 for a fuzzy match"
    assert pytest.approx(results["fuzzy_match"][1], 0.001) == 0.0, "Fuzzy Match should return 0.0 for different letters"
