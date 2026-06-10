"""A metric that evaluates the sustainability of the models by calculating the length of the output text (in words and
tokens) and calculates the energy consumption in Wh as well as the global warming potential in gCO2eq.
"""

import logging
from typing import override

from ecologits.electricity_mix_repository import electricity_mixes
from ecologits.impacts.llm import compute_llm_impacts
from ecologits.utils.range_value import RangeValue

from lmbench.metrics.abstract import LLMAwareMetric
from lmbench.models.data_models import LLMMessage, LLMOutput, LLMRole
from lmbench.task import Task

logger = logging.getLogger(__name__)


class SustainabilityMetrics(LLMAwareMetric):
    """A metric that evaluates the sustainability of a model by calculating global warming potential and energy use.

    This is based on:s
     - the length of the output text in tokens
     - the number of active and total parameters

     Both the sustainability metrics and the number of generated tokens/words are returned.
    """

    name: str = "Sustainability"
    tasks: list[Task] = [Task.SUMMARIZATION, Task.QUESTION_ANSWERING, Task.CLASSIFICATION, Task.TOPIC_EXTRACTION]

    @override
    def description(self) -> str:
        return (
            "Calculates the global warming potential (gCO2eq), the energy consumption (Wh) and the length of the llm"
            "output in tokens, words, and characters."
        )

    @override
    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, list[float]]:
        tokens = []
        words = []
        output_words = []
        reasoning_words = []
        for single_output in model_output:
            answer_only = str(single_output)
            reasoning = ""
            if isinstance(single_output, LLMOutput):
                text = single_output.original_output
                reasoning = single_output.reasoning_output or ""
            else:
                text = single_output

            # This case checks whether we have "secret" reasoning tokens from OpenAI models
            if reasoning.startswith("num_reasoning_tokens:"):
                n_reasoning_tokens = int(reasoning.replace("num_reasoning_tokens:", ""))
                logger.debug(f"Found OpenAI reasoning count {n_reasoning_tokens}")
                n_tokens = self.model.tokenizer.num_tokens([LLMMessage(role=LLMRole.USER, content=answer_only)])

                tokens.append(float(n_reasoning_tokens + n_tokens))
                # Assumes 2 tokens equals 1 word
                words.append(float((n_reasoning_tokens / 2) + len(answer_only.split())))
                output_words.append(float(len(answer_only.split())))
                reasoning_words.append(float(n_reasoning_tokens / 2))
            else:
                # We need to set role to USER, because the huggingface tokenizer will not tokenize a SYSTEM message,
                # if there is no USER message preceding it. In the end, this should not make a difference in token
                # count.
                n_tokens = self.model.tokenizer.num_tokens([LLMMessage(role=LLMRole.USER, content=text)])
                tokens.append(float(n_tokens))

                words.append(float(len(text.split())))
                output_words.append(float(len(answer_only.split())))
                reasoning_words.append(float(len(reasoning.split())))

        europe_energy_mix = electricity_mixes.find_electricity_mix(zone="EEE")
        assert europe_energy_mix is not None, "Cannot find Europe (EEE) electricity mix"
        german_energy_mix = electricity_mixes.find_electricity_mix(zone="DEU")
        assert german_energy_mix is not None, "Cannot find German (DEU) electricity mix"
        active_parameters = self.model.active_parameters if self.model is not None else 0
        total_parameters = self.model.total_parameters if self.model is not None else 0

        gwp_eu = []
        energy = []
        gwp_de = []

        for t in tokens:
            impacts_eu = compute_llm_impacts(
                model_active_parameter_count=active_parameters,
                model_total_parameter_count=total_parameters,
                output_token_count=t,
                if_electricity_mix_adpe=europe_energy_mix.adpe,
                if_electricity_mix_gwp=europe_energy_mix.gwp,
                if_electricity_mix_pe=europe_energy_mix.pe,
            )
            if isinstance(impacts_eu.energy.value, RangeValue):
                energy.append(impacts_eu.energy.value.mean * 1000.0)
            else:
                energy.append(float(impacts_eu.energy.value) * 1000.0)

            if isinstance(impacts_eu.gwp.value, RangeValue):
                gwp_eu.append(impacts_eu.gwp.value.mean * 1000.0)
            else:
                gwp_eu.append(float(impacts_eu.gwp.value) * 1000.0)

            impacts_de = compute_llm_impacts(
                model_active_parameter_count=active_parameters,
                model_total_parameter_count=total_parameters,
                output_token_count=t,
                if_electricity_mix_adpe=german_energy_mix.adpe,
                if_electricity_mix_gwp=german_energy_mix.gwp,
                if_electricity_mix_pe=german_energy_mix.pe,
            )
            if isinstance(impacts_de.gwp.value, RangeValue):
                gwp_de.append(impacts_de.gwp.value.mean * 1000.0)
            else:
                gwp_de.append(float(impacts_de.gwp.value) * 1000.0)

        return {
            "full_tokens": tokens,
            "full_words": words,
            "answer_words": output_words,
            "reasoning_words": reasoning_words,
            "gwp_eu": gwp_eu,
            "energy": energy,
            "gwp_de": gwp_de,
        }
