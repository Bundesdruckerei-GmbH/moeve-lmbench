"""Shared LLM-as-a-judge building blocks for metrics (hallucination, prompt injection, future RAGAS replacements).

- ``lm``: build the dspy judge LM from a ``JudgeLLMConfig``.
- ``calls``: call a judge once or n times with parse retries, and the judge error types.
- ``claims``: claim-F1 (claim extraction and verification against a reference answer).
"""
