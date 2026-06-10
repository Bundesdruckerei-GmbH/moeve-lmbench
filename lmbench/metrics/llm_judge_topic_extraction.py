"""Module providing classes to split and verify topics with an LLM and compute
topic extraction metrics.

This module defines:
    - TopicSplitInput: Pydantic model for topic split input.
    - TopicSplitOutput: Pydantic model for topic split output.
    - TopicSplitPrompt: Prompt to split topic strings.
    - VerifyTopicInput: Pydantic model for topic verification input.
    - VerifyTopicOutput: Pydantic model for single topic verification output.
    - VerifyTopicsOutput: Pydantic model for verification results.
    - VerifyTopicPrompt: Prompt to verify topics.
    - TopicMatch: Metric for topic extraction evaluation using an LLM.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from langchain_core.callbacks import Callbacks
from numpy._typing import NDArray
from pydantic import BaseModel, Field
from ragas import SingleTurnSample
from ragas.metrics import MetricOutputType, MetricType, MetricWithLLM, SingleTurnMetric
from ragas.metrics.utils import fbeta_score
from ragas.prompt import PydanticPrompt


class TopicSplitInput(BaseModel):
    """Pydantic model for topic split prompt input.

    Attributes:
        input_string (str): The string containing topics to split.
    """

    input_string: str = Field(..., description="Input string with topics.")


class TopicSplitOutput(BaseModel):
    """Pydantic model for topic split prompt output.

    Attributes:
        output_list (list[str]): List of topics extracted from input.
    """

    output_list: list[str] = Field(..., description="Output list of topics.")


class TopicSplitPrompt(PydanticPrompt[TopicSplitInput, TopicSplitOutput]):
    """Prompt to split a string into individual topics.

    This prompt instructs the LLM to split an input string with multiple topics
    (delimited by commas or newlines) into a list of topic strings.

    Attributes:
        instruction (str): The instruction template for topic splitting.
        examples (list[tuple]): List of example inputs and outputs.
    """

    instruction = (
        "Split the string into individual topics (separated by commas, newlines, or bullets). "
        "Strip any leading bullets, dashes, numbers or whitespace from each topic. "
        "Do not otherwise alter the wording; return each clean topic as a string."
    )
    input_model = TopicSplitInput
    output_model = TopicSplitOutput
    examples = [
        (
            TopicSplitInput(input_string="economy, healthcare, education"),
            TopicSplitOutput(output_list=["economy", "healthcare", "education"]),
        ),
        (
            TopicSplitInput(input_string="Here are the topics: climate change, renewable energy, carbon taxes"),
            TopicSplitOutput(output_list=["climate change", "renewable energy", "carbon taxes"]),
        ),
        (
            TopicSplitInput(input_string="* Alkylphenole\n* Alkylphenolethoxylate\n* Bisphenol A"),
            TopicSplitOutput(output_list=["Alkylphenole", "Alkylphenolethoxylate", "Bisphenol A"]),
        ),
        (
            TopicSplitInput(input_string="politics\nhealthcare\neducation"),
            TopicSplitOutput(output_list=["politics", "healthcare", "education"]),
        ),
    ]


class VerifyTopicInput(BaseModel):
    """Pydantic model for topic verification prompt input.

    Attributes:
        query_topics (list[str]): Topics that need to be verified.
        reference_topics (list[str]): List of reference topics (already split).
    """

    query_topics: list[str] = Field(
        ...,
        description="List of topics that need to be verified.",
    )
    reference_topics: list[str] = Field(
        ...,
        description="List of reference topics.",
    )


class VerifyTopicOutput(BaseModel):
    """Pydantic model for single topic verification result.

    Attributes:
        reason (str): Reason for verification of the topic.
        verdict (bool): Boolean verdict indicating if the topic was found.
    """

    reason: str = Field(
        ...,
        description="Reason for the verification of the topic.",
    )
    verdict: bool = Field(
        ...,
        description="Boolean verdict indicating if the topic was found among references.",
    )


class VerifyTopicsOutput(BaseModel):
    """Pydantic model for verification results for multiple topics.

    Attributes:
        results (list[VerifyTopicOutput]): List of individual verification outcomes.
    """

    results: list[VerifyTopicOutput] = Field(
        ...,
        description="Verification results for each topic.",
    )


class VerifyTopicPrompt(PydanticPrompt[VerifyTopicInput, VerifyTopicsOutput]):
    """Prompt to verify the presence of query topics in reference topics.

    This prompt instructs the LLM to determine whether each topic in
    `query_topics` appears or is semantically equivalent to any topic
    in `reference_topics`, returning verdicts and reasons.

    Attributes:
        instruction (str): The instruction template for topic verification.
        examples (list[tuple]): List of example inputs and verification outputs.
    """

    instruction = (
        "You are given two lists of topic strings:\n"
        "  • query_topics (the extracted topics to check)\n"
        "  • reference_topics (the gold-standard topics)\n\n"
        "For each query topic:\n"
        "  1. Normalize case, ignore trivial punctuation, and treat plural/singular forms as equivalent.\n"
        "  2. Allow fuzzy and semantic matches: consider shared stems, domain overlap, thematic similarity,\n"
        "     synonyms, and derivational variants.\n"
        "  3. Expand known acronyms when matching.\n"
        "  4. If a query topic broadly or semantically aligns with any reference topic, count it as a match.\n"
        "  5. Use domain knowledge and policy/legal context where applicable.\n"
        "  6. Return true with a concise reason if matched; otherwise false with an explanation.\n\n"
        'Return *only* a JSON object with one key "results", whose value is an array of\n'
        "objects in the same order as query_topics. Each object must have exactly:\n"
        "  {\n"
        '    "reason": <string>,\n'
        '    "verdict": true or false\n'
        "  }\n"
        "Do not include any extra text or formatting."
    )
    input_model = VerifyTopicInput
    output_model = VerifyTopicsOutput
    examples = [
        (
            VerifyTopicInput(
                query_topics=["healthcare", "education"],
                reference_topics=["economy", "healthcare", "education"],
            ),
            VerifyTopicsOutput(
                results=[
                    VerifyTopicOutput(
                        reason="Exact match 'healthcare' found in references.",
                        verdict=True,
                    ),
                    VerifyTopicOutput(
                        reason="Exact match 'education' found in references.",
                        verdict=True,
                    ),
                ],
            ),
        ),
        (
            VerifyTopicInput(
                query_topics=["renewable energy", "carbon taxes"],
                reference_topics=["climate change", "renewable energy", "carbon taxes"],
            ),
            VerifyTopicsOutput(
                results=[
                    VerifyTopicOutput(
                        reason="Exact match 'renewable energy' found in references.",
                        verdict=True,
                    ),
                    VerifyTopicOutput(
                        reason="Exact match 'carbon taxes' found in references.",
                        verdict=True,
                    ),
                ],
            ),
        ),
        (
            VerifyTopicInput(
                query_topics=["economics", "politics"],
                reference_topics=["economy", "education"],
            ),
            VerifyTopicsOutput(
                results=[
                    VerifyTopicOutput(
                        reason="'economics' is semantically similar to 'economy', therefore considered a match.",
                        verdict=True,
                    ),
                    VerifyTopicOutput(
                        reason="'politics' not found among reference topics.",
                        verdict=False,
                    ),
                ],
            ),
        ),
        (
            VerifyTopicInput(
                query_topics=["dogs", "car"],
                reference_topics=["dog", "automobile"],
            ),
            VerifyTopicsOutput(
                results=[
                    VerifyTopicOutput(
                        reason="Plural 'dogs' normalized to 'dog', match found.",
                        verdict=True,
                    ),
                    VerifyTopicOutput(
                        reason="'car' is a synonym of 'automobile', match found.",
                        verdict=True,
                    ),
                ],
            ),
        ),
        (
            VerifyTopicInput(
                query_topics=["Alkylphenolethoxylate"],
                reference_topics=["alkylphenol ethoxylates"],
            ),
            VerifyTopicsOutput(
                results=[
                    VerifyTopicOutput(
                        reason="Minor punctuation and pluralization differences ignored; match found.",
                        verdict=True,
                    ),
                ],
            ),
        ),
        (
            VerifyTopicInput(
                query_topics=["Digitalstrategie", "Zukunftssicherheit"],
                reference_topics=[
                    "Digitale Gesellschaft",
                    "Digitaler Staat",
                    "Digitale Wirtschaft",
                    "Digitale Wissenschaft",
                    "Internationale Datenpolitik",
                    "Schlüsselprojekte",
                ],
            ),
            VerifyTopicsOutput(
                results=[
                    VerifyTopicOutput(
                        reason=(
                            "Shares root ‘Digital-’ and context of strategy in a digital domain; considered a match."
                        ),
                        verdict=True,
                    ),
                    VerifyTopicOutput(
                        reason="No clear thematic or linguistic overlap with any reference topic.",
                        verdict=False,
                    ),
                ],
            ),
        ),
        (
            VerifyTopicInput(
                query_topics=["ErbSTG", "GmbHG-VE"],
                reference_topics=["Erbschaft- und Schenkungsteuerrecht", "Verantwortungseigentum"],
            ),
            VerifyTopicsOutput(
                results=[
                    VerifyTopicOutput(
                        reason="Acronym 'ErbSTG' expanded to 'Erbschaft- und Schenkungsteuerrecht'; match found.",
                        verdict=True,
                    ),
                    VerifyTopicOutput(
                        reason="The suffix 'VE' in 'GmbHG-VE' stands for 'Verantwortungseigentum'; match found.",
                        verdict=True,
                    ),
                ],
            ),
        ),
    ]


@dataclass
class TopicMatch(MetricWithLLM, SingleTurnMetric):
    """Metric for topic extraction.

    This metric evaluates topic extraction by first splitting the model
    response and the reference string into individual topics, then verifies topic matches, and
    computes a precision, recall, or F1 score.

    Attributes:
        name (str): Identifier of the metric.
        _required_columns (dict): Required dataset columns for evaluation.
        output_type (MetricOutputType): Type of the metric output.
        mode (Literal): Scoring mode ('precision', 'recall', or 'f1').
        beta (float): Beta parameter for the F1 score calculation.
        topic_split_prompt (PydanticPrompt): Prompt for splitting topics.
        verify_topic_prompt (PydanticPrompt): Prompt for verifying topics.
        language (str): Language setting for prompts.
    """

    name: str = "topic_match"
    _required_columns: dict[MetricType, set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"response", "reference"}}
    )
    output_type: MetricOutputType | None = MetricOutputType.CONTINUOUS
    mode: Literal["precision", "recall", "f1"] = "f1"
    beta: float = 1.0
    topic_split_prompt: PydanticPrompt = field(default_factory=TopicSplitPrompt)
    verify_topic_prompt: PydanticPrompt = field(default_factory=VerifyTopicPrompt)
    language: str = "english"

    topic_cache: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization to validate that beta is a float."""
        if type(self.beta) is not float:
            raise ValueError(
                "Beta must be a float. A beta > 1 gives more weight to recall, while beta < 1 favors precision."
            )

    async def split_topics(self, topics: str, callbacks: Callbacks) -> list[str]:
        """Split a topic string into individual topics.

        Args:
            topics (str): The input string containing concatenated topics.
            callbacks (Callbacks): Callbacks to be passed to the LLM.

        Returns:
            list[str]: List of extracted topic strings.
        """
        topic_split_input = TopicSplitInput(input_string=topics)
        result = await self.topic_split_prompt.generate(data=topic_split_input, llm=self.llm, callbacks=callbacks)  # type: ignore[reportArgumentType]
        self.topic_cache[topics] = result.output_list
        return result.output_list

    async def verify_topics(
        self, reference_topics: list[str], query_topics: list[str], callbacks: Callbacks
    ) -> NDArray[np.bool_]:
        """Verify query topics against reference topics.

        Args:
            reference_topics (list[str]): List of reference topics to verify against.
            query_topics (list[str]): List of topics that need verification.
            callbacks (Callbacks): Callbacks to be passed to the LLM.

        Returns:
            NDArray[np.bool_]: Boolean array indicating verification verdicts for each query topic.
        """
        verify_topics_input = VerifyTopicInput(query_topics=query_topics, reference_topics=reference_topics)
        response = await self.verify_topic_prompt.generate(data=verify_topics_input, llm=self.llm, callbacks=callbacks)  # type: ignore[reportArgumentType]
        if response.results:
            topic_verification = np.array([result.verdict for result in response.results])
        else:
            topic_verification = np.array([], dtype=bool)
        return topic_verification

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks: Callbacks) -> float:
        """Compute a single-turn score for topic extraction.

        This method extracts topics from both response and reference, computes true
        positives, false positives, and false negatives based on the scoring mode,
        and returns the precision, recall, or F1 score.

        Args:
            sample (SingleTurnSample): Single-turn sample with response and reference.
            callbacks (Callbacks): Callbacks to be passed to the LLM.

        Returns:
            float: Rounded score value (precision, recall, or F1) with two decimal places.
        """
        reference = sample.reference
        response = sample.response
        assert self.llm is not None, "LLM must be set"
        assert reference is not None, "Reference is not set"
        assert response is not None, "Response is not set"

        response_topics = await self.split_topics(response, callbacks)
        reference_topics = await self.split_topics(reference, callbacks)
        reference_response = await self.verify_topics(reference_topics, response_topics, callbacks)

        if self.mode != "precision":
            response_reference = await self.verify_topics(response_topics, reference_topics, callbacks)
        else:
            response_reference = np.array([], dtype=bool)

        tp = sum(reference_response)
        fp = sum(~reference_response)
        if self.mode != "precision":
            fn = sum(~response_reference)
        else:
            fn = 0

        if self.mode == "precision":
            score = tp / (tp + fp + 1e-8)
        elif self.mode == "recall":
            score = tp / (tp + fn + 1e-8)
        else:
            score = fbeta_score(tp, fp, fn, self.beta)

        return np.round(score, 2)
