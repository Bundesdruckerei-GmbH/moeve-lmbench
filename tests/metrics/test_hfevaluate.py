import numpy as np
import pytest

from lmbench.metrics.hfevaluate import BLEU, ROUGE, BERTScore

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


def test_rouge():
    rouge = ROUGE()
    results = rouge.evaluate(model_output, labels)
    assert len(results) == 4, "ROUGE should return four results."
    assert "rouge1" in results, "ROUGE should return rouge1."
    assert "rouge2" in results, "ROUGE should return rouge2."
    assert "rougeL" in results, "ROUGE should return rougeL."
    assert "rougeLsum" in results, "ROUGE should return rougeLsum."

    results = rouge.evaluate(["Das ist ein Test"], ["Das ist ein Test"])
    assert pytest.approx(results["rouge1"], 0.001) == 1.0, "ROUGE1 should return 1.0 for a perfect match."
    assert pytest.approx(results["rouge2"], 0.001) == 1.0, "ROUGE2 should return 1.0 for a perfect match."
    assert pytest.approx(results["rougeL"], 0.001) == 1.0, "ROUGEL should return 1.0 for a perfect match."
    assert pytest.approx(results["rougeLsum"], 0.001) == 1.0, "ROUGELsum should return 1.0 for a perfect match."


def test_bleu():
    bleu = BLEU()
    results = bleu.evaluate(model_output, labels)
    assert len(results) == 1, "BLEU should return one results."
    assert "bleu" in results, "BLEU should return bleu."

    results = bleu.evaluate(["Das ist ein Test"], ["Das ist ein Test"])
    assert pytest.approx(results["bleu"], 0.001) == 1.0, "BLEU should return 1.0 for a perfect match."


def test_bertscore():
    bertscore = BERTScore()
    results = bertscore.evaluate(model_output, labels)
    assert len(results) == 3, "BERTScore should return three results."
    assert "precision" in results, "BERTScore should return precision."
    assert "recall" in results, "BERTScore should return recall."
    assert "f1" in results, "BERTScore should return f1."

    assert np.mean(results["precision"]) < 0.999, "BERTScore precision should be less than 1.0 for a non-perfect."
    assert np.mean(results["recall"]) < 0.999, "BERTScore recall should be less than 1.0 for a non-perfect."
    assert np.mean(results["f1"]) < 0.999, "BERTScore f1 should be less than 1.0 for a non-perfect."

    results = bertscore.evaluate(["Das ist ein Test"], ["Das ist ein Test"])
    assert pytest.approx(results["precision"][0], 0.001) == 1.0, "BERTScore should return 1.0 for a perfect match."
    assert pytest.approx(results["recall"][0], 0.001) == 1.0, "BERTScore should return 1.0 for a perfect match."
    assert pytest.approx(results["f1"][0], 0.001) == 1.0, "BERTScore should return 1.0 for a perfect match."
