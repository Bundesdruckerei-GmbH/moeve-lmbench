# <a href="https://moeve.bundesdruckerei.de/"><img src="docs/assets/moewe_bdr_rgb_72dpi.png" alt="MÖVE" height="26"/></a> MÖVE LMBench

MÖVE LMBench is a benchmarking framework for evaluating Large Language Models across tasks like summarization, question answering, classification, and topic extraction, with metrics covering both task performance and governance criteria such as energy consumption and alignment with German constitutional values. Supported LLM providers include any OpenAI-compatible API (OpenAI, vLLM, LiteLLM, etc.), Azure OpenAI, and Ollama.

It is the evaluation component of **MÖVE** (*Modelle für die Öffentliche Verwaltung Evaluieren*), a benchmark tailored to the German public sector, which combines this framework with German-language datasets reflecting public-administration domains. Up-to-date MÖVE results are published at [moeve.bundesdruckerei.de](https://moeve.bundesdruckerei.de/).

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- An LLM provider (e.g., [Ollama](https://ollama.com/) for local models, or an OpenAI API key)

### Installation

```bash
git clone <repo-url>
cd lmbench
uv sync
```

### Run your first evaluation

The repo includes two sample datasets to get started. Here's how to evaluate a local model using Ollama:

1. Install and start [Ollama](https://ollama.com/)
2. Pull a model: `ollama pull llama3.2:1b`
3. Configure the judge LLM (see [Judge LLM](#judge-llm) below). Some metrics use a secondary LLM to evaluate outputs. Set `OPENAI_API_KEY` in your `.env` file — the default judge uses `gpt-4o-mini` via OpenAI.
4. Run the evaluation:

```bash
uv run lmbench \
    --model "llama3.2:1b" \
    --provider ollama \
    --model_args "ctx_size=128000,num_predict=4096,total_parameters=1" \
    --dataset eurlexsum-german-sample \
    --limit 5
```

Or using an OpenAI-compatible API:

```bash
uv run lmbench \
    --model "gpt-4o-mini" \
    --provider openai \
    --model_args "base_url=https://api.openai.com/v1,api_key=OPENAI_API_KEY,ctx_size=128000,num_predict=16384,total_parameters=1" \
    --dataset german-quad-sample
```

Results are written to the `output/` folder as parquet and CSV files.

> **Note:** Some metrics require additional resources (e.g., the IsGerman metric needs a FastText model). If a metric cannot be initialized, it will be skipped with a warning showing how to set it up.

## Configuration

### Environment Variables

API keys are read from environment variables. Create a `.env` file in the project root (see `.env.example`):

```ini
OPENAI_API_KEY=<your-key>
AZURE_OPENAI_API_KEY=<your-key>
```

The following environment variables can override config paths:

- `DATASET_FOLDER` — where datasets are stored (default: `datasets/`)
- `CACHE_FOLDER` — where the SQLite cache is stored (default: `cache/`)
- `OUTPUT_FOLDER` — where results are written (default: `output/`)
- `TOKENIZER_FOLDER` — where tokenizer files are stored (default: `tokenizer/`)

### config.yaml

The file `lmbench/config/config.yaml` holds default paths and settings. Most users won't need to modify it.

### Judge LLM

Some metrics use a secondary LLM as a judge to evaluate outputs. The judge config in `config.yaml` covers LLM connection settings:

- `judge_llm` — LLM used by ragas-based metrics (RagasQA, RagasComparison, RagasTopicExtraction) and the Values metric.

The `api_key` field is resolved by [OmegaConf](https://omegaconf.readthedocs.io/) using `${oc.env:VAR}` syntax — point it at whichever environment variable holds your key.

**OpenAI (default):**

```yaml
judge_llm:
  provider: openai
  model: gpt-4o-mini
  api_key: "${oc.env:OPENAI_API_KEY}"
```

**OpenAI-compatible API** (vLLM, LiteLLM, Azure AI Foundry `/openai/v1`, etc.):

```yaml
judge_llm:
  provider: openai
  model: my-model
  base_url: http://localhost:8000/v1
  api_key: "${oc.env:MY_API_KEY}"
```

**Azure OpenAI:**

```yaml
judge_llm:
  provider: azure_openai
  model: gpt-4o-mini
  azure_endpoint: https://my-resource.openai.azure.com
  azure_deployment: my-deployment
  api_version: "2024-06-01"
  api_key: "${oc.env:AZURE_OPENAI_API_KEY}"
```

If the judge LLM is misconfigured, ragas-based metrics log `Exception raised in Job[...]` lines at ERROR level and return NaN scores. If you see all-NaN scores or repeated error logs, check that the `judge_llm` config block is reachable.

### Experiment tracking (MLflow)

Each run logs metrics and tags to [MLflow](https://mlflow.org/). By default the tracking destination is the local `./mlruns/` folder, which you can open with `mlflow ui`. Point at a different backend by setting `MLFLOW_TRACKING_URI` before running (e.g. `http://my-mlflow-server:5000` or a `file://` path).

To log to an Azure ML Workspace, install the Azure plugin and set the workspace tracking URI:

```bash
pip install azureml-mlflow
export MLFLOW_TRACKING_URI="azureml://..."
```

## Providers

The `--provider` parameter selects which API backend to use:

| Provider | Description |
|---|---|
| `openai` | Any OpenAI-compatible API (OpenAI, vLLM, LiteLLM, etc.) |
| `azure_openai` | OpenAI models via Azure |
| `ollama` | Local models via Ollama |

### Provider-specific model args

Model args are passed as `--model_args "key1=value1,key2=value2"`.

**Required for all providers:**

| Arg | Description |
|---|---|
| `ctx_size` | Maximum context size in tokens |
| `num_predict` | Maximum output tokens |
| `total_parameters` | Total model parameters (in billions) |

**OpenAI:**

| Arg | Description |
|---|---|
| `base_url` | Base URL of the API (default: OpenAI's API) |
| `api_key` | Name of the environment variable containing the API key |

**Azure OpenAI:**

| Arg | Description |
|---|---|
| `azure_endpoint` | Azure endpoint URL |
| `api_version` | Azure API version |
| `api_key` | Name of the environment variable containing the API key |

**Ollama:**

| Arg | Description |
|---|---|
| `host` | Ollama server URL (default: `http://127.0.0.1:11434`) |

## Datasets

Datasets are stored in the `datasets/` folder. Each dataset is a subfolder containing:

- `config.yaml` — task type, column mappings, and settings
- `dataset.parquet` — the actual data
- `user_prompt.txt` — the prompt template with `{column_name}` placeholders

### Included sample datasets

| Dataset | Task | Source | License |
|---|---|---|---|
| `eurlexsum-german-sample` | Summarization | [EUR-Lex-Sum](https://huggingface.co/datasets/dennlinger/eur-lex-sum) | CC-BY-4.0 |
| `german-quad-sample` | Question Answering | [GermanQuAD](https://huggingface.co/datasets/deepset/germanquad) | CC-BY-4.0 |

These are 10-row samples for testing. See the `LICENSE` file in each dataset folder for full attribution.

### Adding your own dataset

1. Create a folder under `datasets/` with your dataset name
2. Add a `dataset.parquet` file with your data
3. Add a `config.yaml`:
   ```yaml
   task: summarization  # or: question_answering, classification, topic_extraction
   filename: dataset.parquet
   dataset_type: dataframe
   config:
     user_prompt: user_prompt.txt
     target_column: <column with expected output>
     shrink_column: <column with the main input text>
   ```
4. Add a `user_prompt.txt` with placeholders matching your column names, e.g.:
   ```
   Summarize the following text:

   {document}
   ```

## Tokenizer

The `ctx_size` model arg controls the maximum context size. LMBench uses a tokenizer to check that inputs fit within this limit. Available tokenizers (set via `tokenizer` in model args):

- `simple` (default) — approximates 4 tokens per word
- `openai` — uses tiktoken, for OpenAI models
- `huggingface` — uses HuggingFace `AutoTokenizer`, loaded from the `tokenizer/` folder

### Using a HuggingFace tokenizer

Place the tokenizer files (`tokenizer.json`, `tokenizer_config.json`) in a subfolder of `tokenizer/` named after the model or model family. For example, `tokenizer/llama-3.1/` will be used for any model whose name starts with `llama-3.1`.

To pull a tokenizer directly from the HuggingFace Hub, use the helper script:

```bash
uv run python scripts/download_hf_tokenizer.py google/gemma-3-27b-it
```

It downloads only tokenizer-related files into `tokenizer/<derived-name>/` and verifies that they load under the currently pinned `transformers` version. Some newer checkpoints (e.g. Gemma) ship a `tokenizer_config.json` written against transformers 5 conventions and fail to load on the transformers 4.x line that lmbench depends on — for example, `extra_special_tokens` is now expected as a `{name: token}` dict but is sometimes serialized as a bare list. When the script detects such a known transformers-5-vs-4 incompatibility it patches the config in place and retries.

## Caching

LMBench has two layers of cache, both under `CACHE_FOLDER` (default: `cache/`):

- **Model-under-test generation cache** (`cache_v2.sqlite`) — caches responses from the LLM being evaluated.
  - `--no-cache` — disable for a run
  - `--clear-cache` — clear before running
- **Judge LLM cache** (`cache/cache.db`, diskcache) — caches calls made by ragas-based metric judges, always on. There is currently no CLI flag to disable it. Remove the file manually to force a fresh run.

## Testing

```bash
uv run pytest .
```

A few tests instantiate ragas/values metric classes, which build an OpenAI client at construction time and fail if `OPENAI_API_KEY` is unset. Set it in `.env` (any non-empty placeholder works — the test suite doesn't actually call the API) before running the suite.

## Development

```bash
# Install dev dependencies
uv sync

# Linting and formatting
uv run ruff check --fix
uv run ruff format

# Type checking
uv run basedpyright
```

## Docker

The included `Dockerfile` builds a CUDA-enabled image with miniconda, uv, and the lmbench package. No pre-built images are published — build it locally:

```bash
# Base image (CUDA + lmbench)
docker build -t lmbench .

# Run a CLI command
docker run --rm --gpus all -v "$PWD/output:/app/output" lmbench lmbench --help
```

Optional build args select extra runtimes:

| Arg | Effect |
|---|---|
| `CREATE_VLLM=true` | Install vLLM in a separate conda env (`conda run -n vllm ...`) |
| `CREATE_OLLAMA=true` | Install Ollama into the image |
| `TORCH_CPU=true` | Install CPU-only PyTorch (skip CUDA wheels) |
| `BASE_IMAGE=<image>` | Override the base image (default: `nvcr.io/nvidia/cuda:13.3.0-devel-ubuntu22.04`) |

For example, a CPU-only image:

```bash
docker build --build-arg TORCH_CPU=true --build-arg BASE_IMAGE=ubuntu:22.04 -t lmbench-cpu .
```

## Scope of this release

MÖVE LMBench is the evaluation component of MÖVE. The remaining components — orchestration (running all configurations of models × datasets), result export, and the public website — are maintained internally and are not part of this release.

The full set of MÖVE benchmark datasets is **not** part of this release. The results published at [moeve.bundesdruckerei.de](https://moeve.bundesdruckerei.de/) are produced by running this framework against internal datasets we don't redistribute. The two CC-BY-4.0 sample datasets shipped here (`eurlexsum-german-sample`, `german-quad-sample`) exist so you can run the framework end-to-end — bring your own data for actual evaluations.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The included sample datasets are licensed under CC-BY-4.0. See the LICENSE file in each dataset folder for attribution.
