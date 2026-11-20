"""Build the dspy LM for an LLM judge and adapt judge-specific output formats."""

import json
import logging
from typing import Any, override

import dspy
from dspy import Signature
from dspy.adapters.json_adapter import JSONAdapter

from lmbench.config.config import CACHE_FOLDER, JudgeLLMConfig

logger = logging.getLogger(__name__)


def build_judge_lm(judge: JudgeLLMConfig, *, cache: bool = True) -> dspy.LM:
    """Build the dspy LM for a judge config; answers are cached on disk in CACHE_FOLDER/dspy if ``cache``."""
    dspy.configure_cache(disk_cache_dir=str(CACHE_FOLDER / "dspy"))
    lm_kwargs = judge.lm_args.model_dump(exclude_none=True)
    if judge.provider == "azure_openai":
        deployment = judge.azure_deployment or judge.model
        return dspy.LM(
            model=f"azure/{deployment}",
            api_base=f"{judge.azure_endpoint}/openai/deployments/{deployment}",
            api_key=judge.api_key,
            api_version=judge.api_version,
            num_retries=15,
            cache=cache,
            **lm_kwargs,
        )
    return dspy.LM(
        model=f"openai/{judge.model}",
        api_base=judge.base_url,
        api_key=judge.api_key,
        num_retries=15,
        cache=cache,
        **lm_kwargs,
    )


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
