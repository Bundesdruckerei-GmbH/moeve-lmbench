"""Tell model-produced ungradable output apart from a judge failure.

A failed judge score stays 0 only when the model itself produced junk; a judge
failure drops instead. This decides which case a row is from the answer text alone.
"""

from lmbench.models.data_models import LLMOutput

# Reasoning-parser markers that must never survive into a gradable answer.
# Harmony channels are gpt-oss; <think> is used broadly (DeepSeek-R1, Qwen, etc.).
_HARMONY_MARKERS = ("<|start|>", "<|channel|>", "<|message|>", "<|end|>", "assistantfinal")
_THINK_MARKERS = ("<think>", "</think>")


def model_output_is_broken(output: str) -> bool:
    """True when the model — not the judge — produced ungradable output.

    Decided from the stored answer alone, with no judge call: empty or
    whitespace-only text, reasoning with no answer, or leaked reasoning-parser
    markers (`<think>`, Harmony channels) still in the answer.

    Args:
        output: The model's answer text as stored for grading.

    Returns:
        True if the output is model-broken, so its failed score should stay 0.
    """
    parsed = LLMOutput.from_string(output)
    answer = str(parsed).strip()

    if isinstance(parsed, LLMOutput) and parsed.reasoning_output and not answer:
        return True

    if not answer:
        return True

    if any(marker in answer for marker in _THINK_MARKERS):
        return True
    if any(marker in answer for marker in _HARMONY_MARKERS):
        return True

    return False
