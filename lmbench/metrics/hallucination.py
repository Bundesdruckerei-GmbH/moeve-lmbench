"""LLM-as-judge hallucination metric using DSPy."""

import json
import logging
from collections.abc import Callable
from math import nan
from typing import Any, override

import dspy
from dspy import InputField, OutputField, Signature
from dspy.adapters.json_adapter import JSONAdapter
from dspy.primitives import Module
from dspy.primitives.prediction import Prediction
from dspy.utils.exceptions import AdapterParseError
from litellm.exceptions import APIError as LiteLLMAPIError
from tqdm import tqdm

from lmbench.config.config import CACHE_FOLDER, CONFIG
from lmbench.dataset.abstract import Dataset
from lmbench.metrics.abstract import DatasetAwareMetric
from lmbench.metrics.aggregate import NanPolicy
from lmbench.metrics.hallucination_utils import (
    closest_index,
    f1_score,
    majority_bool,
    median_score,
    normalize_hallucination_question_type,
    parse_hallucination_output,
    score_variance,
)
from lmbench.task import Task

logger = logging.getLogger(__name__)


def _build_hallucination_judge_lm() -> dspy.LM:
    assert CONFIG.hallucination_judge_llm is not None, (
        "hallucination_judge_llm config is required for the Hallucination metric."
    )
    dspy.configure_cache(disk_cache_dir=str(CACHE_FOLDER / "dspy"))

    judge = CONFIG.hallucination_judge_llm
    lm_kwargs = judge.lm_args.model_dump(exclude_none=True)
    if judge.provider == "azure_openai":
        deployment = judge.azure_deployment or judge.model
        return dspy.LM(
            model=f"azure/{deployment}",
            api_base=f"{judge.azure_endpoint}/openai/deployments/{deployment}",
            api_key=judge.api_key,
            api_version=judge.api_version,
            num_retries=15,
            **lm_kwargs,
        )
    return dspy.LM(
        model=f"openai/{judge.model}",
        api_base=judge.base_url,
        api_key=judge.api_key,
        num_retries=15,
        **lm_kwargs,
    )


JUDGE_PARSE_RETRIES = 2


class HarmonyUnwrappingJSONAdapter(JSONAdapter):
    """JSONAdapter that strips gpt-oss Harmony ``{"final": "..."}`` envelopes."""

    @override
    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        return super().parse(signature, self._unwrap_harmony(completion))

    @staticmethod
    def _unwrap_harmony(completion: str) -> str:
        if not completion:
            return ""
        try:
            outer = json.loads(completion)
        except json.JSONDecodeError:
            return completion
        if isinstance(outer, dict) and set(outer.keys()) == {"final"}:
            inner = outer["final"]
            unwrapped = inner if isinstance(inner, str) else json.dumps(inner)
            logger.debug("Unwrapped Harmony envelope (inner length=%d)", len(unwrapped))
            return unwrapped
        return completion


def _peek_last_finish_reason() -> str | None:
    """Best-effort finish_reason of the most recent dspy LM call (racy across threads)."""
    try:
        history = dspy.settings.lm.history
        if not history:
            return None
        response = history[-1].get("response")
        choices = getattr(response, "choices", None) or []
        if not choices:
            return None
        return getattr(choices[0], "finish_reason", None)
    except Exception:
        return None


def call_judge_n[JudgeResult](
    evaluator: Callable[..., JudgeResult],
    *,
    n: int,
    temperature: float,
    **kwargs: Any,
) -> list[JudgeResult]:
    """Call a DSPy judge module once or N times, retrying ``AdapterParseError`` per call."""

    def _call_with_retries(rollout_id: int | None) -> JudgeResult:
        last_exc: Exception | None = None
        for attempt in range(JUDGE_PARSE_RETRIES + 1):
            try:
                if rollout_id is None and attempt == 0:
                    return evaluator(**kwargs)
                effective_rollout_id = (rollout_id if rollout_id is not None else 0) + attempt
                effective_temperature = temperature if rollout_id is not None else max(temperature, 1.0)
                return evaluator(
                    **kwargs,
                    config={"rollout_id": effective_rollout_id, "temperature": effective_temperature},
                )
            except AdapterParseError as e:
                last_exc = e
                lm_response = getattr(e, "lm_response", "") or ""
                head = lm_response[:300].replace("\n", "\\n")
                tail = lm_response[-300:].replace("\n", "\\n") if len(lm_response) > 300 else ""
                logger.warning(
                    "Judge parse failure (attempt %d/%d, rollout_id=%s, finish_reason=%s, "
                    "lm_response_len=%d, lm_response_head=%r, lm_response_tail=%r): %s",
                    attempt + 1,
                    JUDGE_PARSE_RETRIES + 1,
                    rollout_id,
                    _peek_last_finish_reason(),
                    len(lm_response),
                    head,
                    tail,
                    str(e).splitlines()[0][:200],
                )
            except LiteLLMAPIError as e:
                logger.warning(
                    "Judge LM API error after dspy retries (rollout_id=%s): %s: %s",
                    rollout_id,
                    type(e).__name__,
                    str(e).splitlines()[0][:200],
                )
                raise
        assert last_exc is not None
        raise last_exc

    if n <= 1:
        return [_call_with_retries(rollout_id=None)]

    results: list[JudgeResult] = []
    last_exc: Exception | None = None
    for shot in range(n):
        try:
            results.append(_call_with_retries(rollout_id=shot * (JUDGE_PARSE_RETRIES + 1)))
        except (AdapterParseError, LiteLLMAPIError) as e:
            last_exc = e
    if not results:
        assert last_exc is not None
        raise last_exc
    if len(results) < n:
        logger.warning(
            "Multi-shot judge call had partial failures: %d/%d rollouts succeeded. "
            "Aggregating over surviving rollouts.",
            len(results),
            n,
        )
    return results


def _parse_bool_verdicts(verdicts, expected_length: int) -> list[bool]:
    """Coerce ``verdicts`` to a ``list[bool]`` of exactly ``expected_length``."""
    if not isinstance(verdicts, list):
        return [False] * expected_length
    bool_verdicts = [bool(v) for v in verdicts]
    if len(bool_verdicts) < expected_length:
        bool_verdicts.extend([False] * (expected_length - len(bool_verdicts)))
    return bool_verdicts[:expected_length]


class StatusCorrectness(Module):
    """Compare the LLM's status output against the expected question type."""

    _STATUS_MAP = {
        "ANSWERABLE": "ANSWERABLE",
        "UNANSWERABLE": "UNANSWERABLE",
        "CONFLICTING": "CONFLICTING",
        "PARSE_ERROR": "PARSE_ERROR",
    }

    def forward(self, llm_status: str, expected_status: str) -> Prediction:
        """Return a Prediction with score 1.0 when the LLM's status matches the expected status."""
        llm_mapped = self._STATUS_MAP.get(llm_status.upper().strip(), llm_status.upper())
        expected_mapped = self._STATUS_MAP.get(expected_status.upper().strip(), expected_status.upper())
        is_correct = llm_mapped == expected_mapped
        return Prediction(
            score=1.0 if is_correct else 0.0,
            is_correct=is_correct,
            reasoning=(f"'{llm_status}' {'matches' if is_correct else 'does not match'} '{expected_status}'"),
        )


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
                sum(_parse_bool_verdicts(r.verdicts, len(llm_claims))) / len(llm_claims) for r in p_results
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
                sum(_parse_bool_verdicts(r.verdicts, len(gold_claims))) / len(gold_claims) for r in r_results
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
            _parse_bool_verdicts(p_results[representative_idx].verdicts, len(llm_claims))
            if llm_claims and p_results
            else []
        )
        rep_r_verdicts = (
            _parse_bool_verdicts(r_results[representative_idx].verdicts, len(gold_claims))
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


class DocumentReferenceF1(Module):
    """Compute set-based F1 between cited and gold document references."""

    def forward(self, llm_documents: list[int], gold_documents: list[int]) -> Prediction:
        """Return precision/recall/F1 of the document reference overlap."""
        if len(gold_documents) == 0:
            p, r = (1.0 if not llm_documents else 0.0), 1.0
            reasoning = "No documents expected"
        elif len(llm_documents) == 0:
            p = r = 0.0
            reasoning = "LLM provided no document references"
        else:
            llm_set, gold_set = set(llm_documents), set(gold_documents)
            inter = llm_set & gold_set
            p = len(inter) / len(llm_set)
            r = len(inter) / len(gold_set)
            reasoning = f"LLM cited {len(llm_set)}, gold {len(gold_set)}, overlap {len(inter)}"
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return Prediction(score=f1, precision=p, recall=r, f1_score=f1, reasoning=reasoning)


class ContextGroundedness(Module):
    """Compute context groundedness via claim extraction + binary verification (RAGAS-style).

    1. Extract atomic claims from the LLM answer.
    2. For each claim, verify whether it is grounded in the context (binary).
    3. Groundedness = number of grounded claims / total claims.
    """

    def __init__(self, n: int = 1, temperature: float = 0.3):
        """Initialise claim extractor and groundedness verifier with multi-shot config."""
        super().__init__()
        self.claim_extractor = dspy.ChainOfThought(ClaimExtractionSignature)
        self.groundedness_verifier = dspy.ChainOfThought(ClaimSupportVerificationSignature)
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

    def forward(self, question: str, context_documents: str, llm_answer: str) -> Prediction:
        """Compute the fraction of LLM answer claims that are grounded in the context."""
        # Step 1: Extract claims from LLM answer
        claims = self._extract_claims(llm_answer, question)

        if not claims:
            return Prediction(
                score=1.0,
                groundedness=1.0,
                claims=[],
                verdicts=[],
                judge_score_variance=0.0,
            )

        # Step 2: Verify groundedness n times
        results = call_judge_n(
            self.groundedness_verifier,
            n=self.n,
            temperature=self.temperature,
            claims=claims,
            reference=context_documents,
            question=question,
        )

        scores = [sum(_parse_bool_verdicts(r.verdicts, len(claims))) / len(claims) for r in results]

        # Step 3: Final score (median across n runs)
        score = median_score(scores)
        representative_idx = closest_index(scores, score)
        rep_verdicts = _parse_bool_verdicts(results[representative_idx].verdicts, len(claims))

        return Prediction(
            score=score,
            groundedness=score,
            claims=claims,
            verdicts=rep_verdicts,
            judge_score_variance=score_variance(scores),
        )


class RefusalCheck(dspy.Signature):
    """Evaluate whether the LLM refrains from giving a single direct answer to the question.
    Answer True if the LLM does NOT commit to one definitive answer.
    Answer False if the LLM answers the question as though no conflict exists.
    """

    question: str = InputField(desc="The original question")
    context_documents: str = InputField(desc="The context documents provided to the LLM")
    llm_response: str = InputField(desc="The LLM's response")
    has_refused: bool = OutputField(desc="True = did not commit to a single answer")


class UnanswerabilityAcknowledgement(dspy.Signature):
    """Evaluate whether the LLM explicitly states that the question cannot be answered from
    the given context. Look for phrases like 'cannot be determined',
    'not enough information', 'the documents do not mention', or equivalent.
    """

    question: str = InputField(desc="The original question")
    context_documents: str = InputField(desc="The context documents provided to the LLM")
    llm_response: str = InputField(desc="The LLM's response")
    states_unanswerable: bool = OutputField(desc="True = explicitly says the question cannot be answered from context")


class ParametricKnowledgeAbsence(dspy.Signature):
    """Evaluate the LLM's use of internal/world knowledge when answering.
    Answer False if the response contains claims that do NOT appear
    in the provided context documents — even if hedged with 'I think' or
    'generally speaking'. Answer True only if ALL factual content in the
    response is either from the context or is about the absence of information.
    """

    question: str = InputField(desc="The original question")
    context_documents: str = InputField(desc="The context documents provided to the LLM")
    llm_response: str = InputField(desc="The LLM's response")
    avoids_parametric: bool = OutputField(desc="True = does not inject external knowledge to answer")


class RefusalQuality(Module):
    """Score the quality of a refusal as the average of three binary judgments."""

    def __init__(self, n: int = 1, temperature: float = 0.3):
        """Initialise the three binary sub-evaluators with multi-shot config."""
        super().__init__()
        self.refusal_evaluator = dspy.ChainOfThought(RefusalCheck)
        self.unanswerability_evaluator = dspy.ChainOfThought(UnanswerabilityAcknowledgement)
        self.knowledge_leakage_evaluator = dspy.ChainOfThought(ParametricKnowledgeAbsence)
        self.n = n
        self.temperature = temperature

    def forward(self, question: str, context_documents: str, llm_response: str) -> Prediction:
        """Aggregate refusal, unanswerability acknowledgement, and parametric-knowledge-absence votes."""
        refusal_results = call_judge_n(
            self.refusal_evaluator,
            n=self.n,
            temperature=self.temperature,
            question=question,
            context_documents=context_documents,
            llm_response=llm_response,
        )
        acknowledgement_results = call_judge_n(
            self.unanswerability_evaluator,
            n=self.n,
            temperature=self.temperature,
            question=question,
            context_documents=context_documents,
            llm_response=llm_response,
        )
        knowledge_absence_results = call_judge_n(
            self.knowledge_leakage_evaluator,
            n=self.n,
            temperature=self.temperature,
            question=question,
            context_documents=context_documents,
            llm_response=llm_response,
        )

        refusals = [result.has_refused for result in refusal_results]
        acknowledgements = [result.states_unanswerable for result in acknowledgement_results]
        knowledge_absences = [result.avoids_parametric for result in knowledge_absence_results]
        sample_scores = [sum(values) / 3 for values in zip(refusals, acknowledgements, knowledge_absences)]
        refusal = majority_bool(refusals)
        acknowledgement = majority_bool(acknowledgements)
        knowledge_absence = majority_bool(knowledge_absences)
        score = sum([refusal, acknowledgement, knowledge_absence]) / 3

        return Prediction(
            score=score,
            overall_quality=score,
            refusal_clarity=refusal,
            explanation_quality=acknowledgement,
            knowledge_usage_check=knowledge_absence,
            judge_score_variance=score_variance(sample_scores),
        )


class KnowledgeLeakageSignature(Signature):
    """Detect if the LLM inappropriately used its internal knowledge to answer an unanswerable question."""

    question: str = InputField(desc="The unanswerable question")
    context_documents: str = InputField(desc="The context (missing the answer)")
    llm_response: str = InputField(desc="The LLM's response")

    contains_answer: bool = OutputField(desc="Does the response contain an actual answer to the question?")
    evidence_of_leakage: str = OutputField(desc="Evidence that LLM used internal knowledge not present in context")
    leakage_detected: bool = OutputField(desc="True if knowledge leakage is detected, False otherwise")


class KnowledgeLeakage(Module):
    """Detect whether the LLM injected external knowledge on unanswerable questions."""

    def __init__(self, n: int = 1, temperature: float = 0.3):
        """Initialise the knowledge-leakage evaluator with multi-shot config."""
        super().__init__()
        self.evaluator = dspy.ChainOfThought(KnowledgeLeakageSignature)
        self.n = n
        self.temperature = temperature

    def forward(self, question: str, context_documents: str, llm_response: str) -> Prediction:
        """Return 1.0 when no leakage is detected (majority vote across runs), 0.0 otherwise."""
        results = call_judge_n(
            self.evaluator,
            n=self.n,
            temperature=self.temperature,
            question=question,
            context_documents=context_documents,
            llm_response=llm_response,
        )
        leakage_votes = [result.leakage_detected for result in results]
        sample_scores = [0.0 if leakage_detected else 1.0 for leakage_detected in leakage_votes]
        leakage_detected = majority_bool(leakage_votes)
        contains_answer = majority_bool([result.contains_answer for result in results])
        score = 0.0 if leakage_detected else 1.0
        representative = results[closest_index(sample_scores, score)]
        return Prediction(
            score=score,
            leakage_detected=leakage_detected,
            contains_answer=contains_answer,
            evidence_of_leakage=representative.evidence_of_leakage,
            judge_score_variance=score_variance(sample_scores),
        )


class ConflictAcknowledgment(dspy.Signature):
    """Evaluate whether the LLM explicitly state that the sources contain conflicting or
    contradictory information. Look for words like 'conflict', 'contradict',
    'disagree', 'inconsistent', or equivalent phrasing.
    """

    question: str = InputField(desc="The original question")
    llm_response: str = InputField(desc="The LLM's response")
    acknowledges_conflict: bool = OutputField(desc="True = explicitly mentions a conflict/contradiction")


class ConflictNeutrality(dspy.Signature):
    """Evaluate whether the LLM remain neutral. Answer False if it picks one claim as correct,
    endorses one source over the other, or dismisses one side. Minor hedging
    ('seems more likely') still counts as NOT neutral.
    """

    question: str = InputField(desc="The original question")
    context_documents: str = InputField(desc="Numbered context documents")
    llm_response: str = InputField(desc="The LLM's response")
    gold_documents: list[int] = InputField(desc="Document numbers that conflict")
    is_neutral: bool = OutputField(desc="True = does not favor either conflicting claim")


class ConflictDetectionQuality(Module):
    """Score conflict handling as the average of refusal, acknowledgement, and neutrality votes."""

    def __init__(self, n: int = 1, temperature: float = 0.3):
        """Initialise the three sub-evaluators with multi-shot config."""
        super().__init__()
        self.conflict_refusal_evaluator = dspy.ChainOfThought(RefusalCheck)
        self.conflict_acknowledge_evaluator = dspy.ChainOfThought(ConflictAcknowledgment)
        self.conflict_neutrality_evaluator = dspy.ChainOfThought(ConflictNeutrality)
        self.n = n
        self.temperature = temperature

    def forward(
        self,
        question: str,
        context_documents: str,
        llm_response: str,
        gold_documents: list[int],
    ) -> Prediction:
        """Aggregate refusal, conflict acknowledgement, and neutrality majority votes."""
        refusal_results = call_judge_n(
            self.conflict_refusal_evaluator,
            n=self.n,
            temperature=self.temperature,
            question=question,
            context_documents=context_documents,
            llm_response=llm_response,
        )
        acknowledgement_results = call_judge_n(
            self.conflict_acknowledge_evaluator,
            n=self.n,
            temperature=self.temperature,
            question=question,
            llm_response=llm_response,
        )
        neutrality_results = call_judge_n(
            self.conflict_neutrality_evaluator,
            n=self.n,
            temperature=self.temperature,
            question=question,
            context_documents=context_documents,
            llm_response=llm_response,
            gold_documents=gold_documents,
        )

        refusals = [result.has_refused for result in refusal_results]
        acknowledgements = [result.acknowledges_conflict for result in acknowledgement_results]
        neutralities = [result.is_neutral for result in neutrality_results]
        sample_scores = [sum(values) / 3 for values in zip(refusals, acknowledgements, neutralities)]
        refusal = majority_bool(refusals)
        acknowledgement = majority_bool(acknowledgements)
        neutrality = majority_bool(neutralities)
        score = sum([refusal, acknowledgement, neutrality]) / 3

        return Prediction(
            score=score,
            overall_quality=score,
            conflict_recognition=refusal,
            explanation_quality=acknowledgement,
            neutrality_check=neutrality,
            judge_score_variance=score_variance(sample_scores),
        )


class HallucinationMetric(DatasetAwareMetric):
    """LLM-as-judge benchmark combining sub-metrics per question type."""

    name: str = "Hallucination"
    tasks: list[Task] = [Task.HALLUCINATION]
    nan_policy: NanPolicy = NanPolicy.OMIT

    def __init__(self, dataset: Dataset, selected_indices: list[int] | None = None) -> None:
        """Instantiate the judge LM and per-question-type sub-evaluators."""
        super().__init__(dataset, selected_indices=selected_indices)
        assert CONFIG.hallucination_judge_llm is not None, (
            "hallucination_judge_llm config is required for the Hallucination metric."
        )
        self.config = CONFIG.hallucination
        dspy.configure(lm=_build_hallucination_judge_lm(), adapter=HarmonyUnwrappingJSONAdapter())
        judge_n = self.config.judge_n
        judge_temperature = CONFIG.hallucination_judge_llm.lm_args.temperature or 0.0
        if judge_n > 1 and judge_temperature == 0.0:
            logger.warning(
                "judge_n=%d with lm_args.temperature=0 would produce identical rollouts "
                "(rollout_id only varies generation at T>0). Using temperature=1.0 for "
                "multi-shot diversity.",
                judge_n,
            )
            judge_temperature = 1.0
        self._status_correctness = StatusCorrectness()
        self._factual_correctness = FactualCorrectnessF1(n=judge_n, temperature=judge_temperature)
        self._document_accuracy = DocumentReferenceF1()
        self._context_groundedness = ContextGroundedness(n=judge_n, temperature=judge_temperature)
        self._refusal_quality = RefusalQuality(n=judge_n, temperature=judge_temperature)
        self._knowledge_leakage = KnowledgeLeakage(n=judge_n, temperature=judge_temperature)
        self._conflict_detection_quality = ConflictDetectionQuality(n=judge_n, temperature=judge_temperature)

    @override
    def description(self) -> str:
        return "LLM-as-judge hallucination benchmark score. Higher is better."

    @override
    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, list[float]]:
        questions = self.get_selected_column("question")
        question_types = self.get_selected_column("question_type")
        first_documents = self.get_selected_column("first_document")
        second_documents = self.get_selected_column("second_document")
        third_documents = self.get_selected_column("third_document")
        gold_answers = self.get_selected_column("gold_answer")
        gold_references = self.get_selected_column("references")

        if len(model_output) != len(labels):
            raise ValueError(
                "HallucinationMetric expected the same number of model outputs and labels, "
                f"got {len(model_output)} outputs and {len(labels)} labels."
            )
        if len(model_output) != len(questions):
            raise ValueError(
                "HallucinationMetric requires one model output per dataset row. "
                f"Got {len(model_output)} model outputs for a dataset with {len(questions)} rows."
            )

        result: dict[str, list[float]] = {
            "hallucination_benchmark_score": [],
            "status_accuracy": [],
            "factual_correctness": [],
            "reference_f1": [],
            "context_groundedness": [],
            "refusal_quality": [],
            "knowledge_leakage_avoidance": [],
            "conflict_handling": [],
        }
        evaluator_result: dict[str, list[float]] = {
            "evaluator_judge_self_consistency_variance": [],
            "evaluator_factual_correctness_variance": [],
            "evaluator_context_groundedness_variance": [],
            "evaluator_refusal_quality_variance": [],
            "evaluator_knowledge_leakage_avoidance_variance": [],
            "evaluator_conflict_handling_variance": [],
        }

        samples = zip(
            questions,
            question_types,
            gold_references,
            first_documents,
            second_documents,
            third_documents,
            model_output,
            gold_answers,
        )
        judge_failure_count = 0
        for q, t, r, d1, d2, d3, a, ga in tqdm(
            samples,
            total=len(model_output),
            desc="Evaluating hallucination",
        ):
            gold_refs: list[int] = [int(x) for x in r]
            doc_context = "\n\n".join(
                f"[Document {i + 1}]\n{doc}" for i, doc in enumerate([d1, d2, d3]) if doc and doc.strip()
            )

            parsed_response = parse_hallucination_output(a)
            llm_status = parsed_response.status
            llm_refs = parsed_response.documents
            llm_response = parsed_response.response_text

            qtype = normalize_hallucination_question_type(t)

            row_scores: dict[str, float] = {
                "status_accuracy": nan,
                "factual_correctness": nan,
                "reference_f1": nan,
                "context_groundedness": nan,
                "refusal_quality": nan,
                "knowledge_leakage_avoidance": nan,
                "conflict_handling": nan,
                "hallucination_benchmark_score": nan,
            }
            row_evaluator_variances: dict[str, float] = {
                "evaluator_judge_self_consistency_variance": nan,
                "evaluator_factual_correctness_variance": nan,
                "evaluator_context_groundedness_variance": nan,
                "evaluator_refusal_quality_variance": nan,
                "evaluator_knowledge_leakage_avoidance_variance": nan,
                "evaluator_conflict_handling_variance": nan,
            }

            try:
                row_scores["status_accuracy"] = self._status_correctness(
                    llm_status=llm_status.value,
                    expected_status=qtype,
                ).score
                sample_score: dict[str, float] = {"status_correctness": row_scores["status_accuracy"]}
                sample_score_variance: dict[str, float] = {}

                if qtype == "ANSWERABLE":
                    factual_correctness = self._factual_correctness(
                        question=q,
                        gold_answer=ga,
                        llm_answer=llm_response,
                        context_documents=doc_context,
                    )
                    sample_score["factual_correctness"] = factual_correctness.score
                    row_scores["factual_correctness"] = factual_correctness.score
                    sample_score_variance["factual_correctness"] = factual_correctness.judge_score_variance
                    row_evaluator_variances["evaluator_factual_correctness_variance"] = (
                        factual_correctness.judge_score_variance
                    )
                    sample_score["document_f1"] = self._document_accuracy(
                        llm_documents=llm_refs, gold_documents=gold_refs
                    ).score
                    row_scores["reference_f1"] = sample_score["document_f1"]
                    context_groundedness = self._context_groundedness(
                        question=q,
                        context_documents=doc_context,
                        llm_answer=llm_response,
                    )
                    sample_score["context_groundedness"] = context_groundedness.score
                    row_scores["context_groundedness"] = context_groundedness.score
                    sample_score_variance["context_groundedness"] = context_groundedness.judge_score_variance
                    row_evaluator_variances["evaluator_context_groundedness_variance"] = (
                        context_groundedness.judge_score_variance
                    )
                    weighted_score = sum(
                        sample_score.get(k, 0.0) * w for k, w in self.config.answerable_weights.items()
                    )
                    weighted_judge_variance = sum(
                        sample_score_variance.get(k, 0.0) * w for k, w in self.config.answerable_weights.items()
                    )
                elif qtype == "UNANSWERABLE":
                    refusal_quality = self._refusal_quality(
                        question=q,
                        context_documents=doc_context,
                        llm_response=llm_response,
                    )
                    sample_score["refusal_quality"] = refusal_quality.score
                    row_scores["refusal_quality"] = refusal_quality.score
                    sample_score_variance["refusal_quality"] = refusal_quality.judge_score_variance
                    row_evaluator_variances["evaluator_refusal_quality_variance"] = refusal_quality.judge_score_variance
                    knowledge_leakage = self._knowledge_leakage(
                        question=q,
                        context_documents=doc_context,
                        llm_response=llm_response,
                    )
                    sample_score["knowledge_leakage"] = knowledge_leakage.score
                    row_scores["knowledge_leakage_avoidance"] = knowledge_leakage.score
                    sample_score_variance["knowledge_leakage"] = knowledge_leakage.judge_score_variance
                    row_evaluator_variances["evaluator_knowledge_leakage_avoidance_variance"] = (
                        knowledge_leakage.judge_score_variance
                    )
                    weighted_score = sum(
                        sample_score.get(k, 0.0) * w for k, w in self.config.unanswerable_weights.items()
                    )
                    weighted_judge_variance = sum(
                        sample_score_variance.get(k, 0.0) * w for k, w in self.config.unanswerable_weights.items()
                    )
                elif qtype == "CONFLICTING":
                    conflict_detection_quality = self._conflict_detection_quality(
                        question=q,
                        context_documents=doc_context,
                        llm_response=llm_response,
                        gold_documents=gold_refs,
                    )
                    sample_score["conflict_detection_quality"] = conflict_detection_quality.score
                    row_scores["conflict_handling"] = conflict_detection_quality.score
                    sample_score_variance["conflict_detection_quality"] = (
                        conflict_detection_quality.judge_score_variance
                    )
                    row_evaluator_variances["evaluator_conflict_handling_variance"] = (
                        conflict_detection_quality.judge_score_variance
                    )
                    sample_score["document_f1"] = self._document_accuracy(
                        llm_documents=llm_refs, gold_documents=gold_refs
                    ).score
                    row_scores["reference_f1"] = sample_score["document_f1"]
                    weighted_score = sum(
                        sample_score.get(k, 0.0) * w for k, w in self.config.conflicting_weights.items()
                    )
                    weighted_judge_variance = sum(
                        sample_score_variance.get(k, 0.0) * w for k, w in self.config.conflicting_weights.items()
                    )
                else:
                    logger.warning("Unknown question type %r for question %r. Skipping detailed evaluation.", qtype, q)
                    weighted_score = sample_score["status_correctness"]  # Fallback to just status correctness
                    weighted_judge_variance = 0.0

                row_scores["hallucination_benchmark_score"] = weighted_score
                row_evaluator_variances["evaluator_judge_self_consistency_variance"] = weighted_judge_variance
            except (AdapterParseError, LiteLLMAPIError) as e:
                judge_failure_count += 1
                logger.error(
                    "Hallucination metric failed on row (question=%r) after retries (%s): %s. Recording NaN scores.",
                    q[:60],
                    type(e).__name__,
                    str(e).splitlines()[0][:200],
                )

            for key, value in row_scores.items():
                result[key].append(value)
            for key, value in row_evaluator_variances.items():
                evaluator_result[key].append(value)

        if self.config.judge_n > 1:
            result.update(evaluator_result)

        result["judge_failure_count"] = judge_failure_count  # type: ignore[reportAssignmentType]

        return result
