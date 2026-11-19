# Changelog

All notable changes to LMBench are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-11-18

### Added

- **Per-item score artifact** — every run writes `per_item_scores.parquet` with one
  score row per item and metric, the input for computing confidence intervals on the
  aggregates. Rows carry the clusters and strata to resample by, and each published
  aggregate can be recomputed from them. Uploaded to MLflow; `--log-run-artifacts`
  (default on) uploads the run's other outputs with it.
- **Content-filter handling** — when OpenAI, Azure OpenAI or Anthropic reject a
  request or filter an answer, the item is recorded as filtered instead of being
  retried and aborting the run.
- Per-generation scores for the Values metric, behind its reported proportions.
- An optional `generation_index` dataset column to mark repeated samples of one item.

### Changed

- **Datasets need an `item_id` column** with a stable, unique id per row; runs on a
  dataset without one fail. The bundled sample datasets are stamped. See "Adding your
  own dataset" in the README.
- **Ragas metrics no longer score judge failures as 0.** A NaN caused by a failed
  judge call is left out of the mean; empty or broken model output still scores 0.
  Results only differ from earlier releases where a judge call failed.
- Datasets with `n_iterations > 1` no longer require `--no-cache` and
  `--temperature > 0`; each repeat is cached separately.
- Hallucination metric: sub-scores are reported under their published names
  (`status_accuracy`, `reference_f1`, `conflict_handling`, ...), and the config weight
  `knowledge_leakage` is renamed to `knowledge_leakage_avoidance`. Configs that set
  the old key need updating.
- Parquet I/O uses pyarrow instead of fastparquet.
- Docker base image updated to CUDA 13.5.0; dependencies refreshed, including fixes
  for published security advisories.

### Fixed

- Multi-threaded runs return generations in dataset order.

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
