"""Claim-F1: split answers into atomic claims and verify them against a reference with an LLM judge."""

import dspy
from dspy import InputField, OutputField, Signature
from dspy.primitives import Module
from dspy.primitives.prediction import Prediction

from lmbench.metrics.judge.calls import call_judge_n
from lmbench.metrics.utils import closest_index, f1_score, median_score, score_variance


def parse_bool_verdicts(verdicts, expected_length: int) -> list[bool]:
    """Coerce ``verdicts`` to a ``list[bool]`` of exactly ``expected_length``."""
    if not isinstance(verdicts, list):
        return [False] * expected_length
    bool_verdicts = [bool(v) for v in verdicts]
    if len(bool_verdicts) < expected_length:
        bool_verdicts.extend([False] * (expected_length - len(bool_verdicts)))
    return bool_verdicts[:expected_length]


class ClaimExtractionSignature(Signature):
    """Extract all atomic factual claims from the given text. Each claim must be a
    single, self-contained factual statement that can be independently verified.
    Do NOT include opinions, questions, or hedged/uncertain statements.
    """

    text: str = InputField(desc="The text to extract claims from")
    question: str = InputField(desc="The question being answered, for context")
    claims: list[str] = OutputField(
        desc="List of atomic factual claims extracted from the text, each a complete sentence"
    )


class ClaimSupportVerificationSignature(Signature):
    """For each claim, determine whether it is supported by the reference text.
    A claim is supported if the reference text explicitly states or logically entails
    the information in the claim. Be strict: if the reference does not contain
    sufficient information to verify a claim, mark it as NOT supported.
    """

    claims: list[str] = InputField(desc="List of claims to verify, in order")
    reference: str = InputField(desc="The reference text to verify claims against")
    question: str = InputField(desc="The question being answered, for context")
    verdicts: list[bool] = OutputField(
        desc="For each claim (same order): True if supported/entailed by the reference, False otherwise"
    )


class FactualCorrectnessF1(Module):
    """Claim-level F1 of LLM answer vs gold answer; precision uses gold+context."""

    def __init__(self, n: int = 1, temperature: float = 0.3):
        """Initialise claim extractor and verifier with multi-shot config."""
        super().__init__()
        self.claim_extractor = dspy.ChainOfThought(ClaimExtractionSignature)
        self.claim_verifier = dspy.ChainOfThought(ClaimSupportVerificationSignature)
        self.n = n
        self.temperature = temperature

    def _extract_claims(self, text: str, question: str) -> list[str]:
        """Extract claims from the given text."""
        results = call_judge_n(
            self.claim_extractor,
            n=1,
            temperature=self.temperature,
            text=text,
            question=question,
        )
        claims = results[0].claims
        if not isinstance(claims, list):
            return []
        return [c for c in claims if isinstance(c, str) and c.strip()]

    def forward(self, question: str, gold_answer: str, llm_answer: str, context_documents: str) -> Prediction:
        """Compute precision/recall/F1 via claim extraction and binary verification."""
        # Step 1: Extract claims from both answers (deterministic, once)
        gold_claims = self._extract_claims(gold_answer, question)
        llm_claims = self._extract_claims(llm_answer, question)

        # Edge case: both empty
        if not gold_claims and not llm_claims:
            return Prediction(
                score=1.0,
                precision=1.0,
                recall=1.0,
                f1_score=1.0,
                gold_claims=[],
                llm_claims=[],
                precision_verdicts=[],
                recall_verdicts=[],
                judge_score_variance=0.0,
            )

        # Step 2: Verify claims n times for robustness
        # --- Precision (LLM claims vs gold answer + context) ---
        if llm_claims:
            p_results = call_judge_n(
                self.claim_verifier,
                n=self.n,
                temperature=self.temperature,
                claims=llm_claims,
                reference=context_documents + "\n\n" + gold_answer,
                question=question,
            )
            sample_precisions = [
                sum(parse_bool_verdicts(r.verdicts, len(llm_claims))) / len(llm_claims) for r in p_results
            ]
        else:
            p_results = []
            # No LLM claims: precise but recall will penalize
            sample_precisions = [1.0] * self.n

        # --- Recall (gold claims vs LLM answer) ---
        if gold_claims:
            r_results = call_judge_n(
                self.claim_verifier,
                n=self.n,
                temperature=self.temperature,
                claims=gold_claims,
                reference=llm_answer,
                question=question,
            )
            sample_recalls = [
                sum(parse_bool_verdicts(r.verdicts, len(gold_claims))) / len(gold_claims) for r in r_results
            ]
        else:
            r_results = []
            sample_recalls = [1.0] * self.n

        # Step 3: Compute final scores
        sample_f1s = [f1_score(p, r) for p, r in zip(sample_precisions, sample_recalls)]
        final_p = median_score(sample_precisions)
        final_r = median_score(sample_recalls)
        final_f1 = f1_score(final_p, final_r)

        # Pick representative verdicts from the run closest to median F1
        representative_idx = closest_index(sample_f1s, final_f1)
        rep_p_verdicts = (
            parse_bool_verdicts(p_results[representative_idx].verdicts, len(llm_claims))
            if llm_claims and p_results
            else []
        )
        rep_r_verdicts = (
            parse_bool_verdicts(r_results[representative_idx].verdicts, len(gold_claims))
            if gold_claims and r_results
            else []
        )

        return Prediction(
            score=final_f1,
            precision=final_p,
            recall=final_r,
            f1_score=final_f1,
            gold_claims=gold_claims,
            llm_claims=llm_claims,
            precision_verdicts=rep_p_verdicts,
            recall_verdicts=rep_r_verdicts,
            judge_score_variance=score_variance(sample_f1s),
        )
