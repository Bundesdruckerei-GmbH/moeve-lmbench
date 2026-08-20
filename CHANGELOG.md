# Changelog

All notable changes to LMBench are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-08-20

### Added

- **Hallucination metric** (dataset `task: hallucination`) — an LLM-as-judge metric built
  on dspy that scores generated answers against retrieved documents. It splits
  evaluation by question type (answerable, unanswerable, conflicting) and reports
  weighted sub-metrics: status correctness, factual correctness, document F1,
  context groundedness, refusal quality, knowledge leakage, and conflict
  detection quality. Configured via the `hallucination_judge_llm` and
  `hallucination` blocks; multi-shot judging (`judge_n`) additionally reports
  per-sub-metric score variance. Contributed by Fraunhofer AISec.
- **Anthropic provider** (`--provider anthropic`) — Claude models via the
  Anthropic Messages API, with a matching tokenizer backed by
  `messages.count_tokens`. Configured with `ANTHROPIC_API_KEY`, and optionally
  `ANTHROPIC_BASE_URL` for gateway routing.
- `judge_failure_count` outputs for ragas metrics — one count per sub-metric, so
  NaN scores caused by judge failures are visible instead of silently averaged away.
- Support for multiple shrink columns: `shrink_column` now accepts a list, with a
  per-column shrink factor applied in the configured order.
- An optional `seed` in the judge LM args for more reproducible judging.
- `scripts/download_hf_tokenizer.py`, which downloads and patches HuggingFace
  tokenizers for offline use.

### Changed

- MLflow now defaults to the local file store (`file:./mlruns`), pinned because
  mlflow 3.13's default SQLite backend is unusable under mlflow-skinny. An
  externally set `MLFLOW_TRACKING_URI` still wins.
- Retry handling is configurable via `max_retries`, with more conservative
  defaults for long-running evaluations.
- Docker base image updated to CUDA 13.3.1; dependencies refreshed.

### Fixed

- The published `uv.lock` no longer disagrees with `pyproject.toml`. Previous
  releases shipped a lockfile that still declared the Azure packages, so `uv sync`
  silently rewrote it.

## [0.2.0] - 2026-05-26

Initial public release of LMBench, a benchmarking framework developed
internally at BDR to evaluate Large Language Models for public-sector
applications. The framework is published to provide transparency for the
evaluation results released alongside it on the BDR website.

The internal datasets used to produce those published results are not part
of this repository; two small public-license sample datasets (EUR-Lex-Sum,
GermanQuAD) are included for getting started.

The pre-1.0 version signals that CLI flags, config schema, and plugin
registry keys may still change between minor versions. Subsequent entries
will follow the Keep-a-Changelog Added / Changed / Fixed / Removed structure.
