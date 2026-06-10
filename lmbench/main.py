"""The CLI of LMbench."""

import argparse
import logging
import logging.config
from logging.config import dictConfig

import mlflow

from lmbench.config.log_conf import log_config
from lmbench.dataset.dataset_utils import load_dataset
from lmbench.dataset.multi_iteration import MultiIterationDataset
from lmbench.evaluation import EvaluationRunner
from lmbench.models import MODEL_MAPPING
from lmbench.utils import quantization_from_model

dictConfig(log_config)
logger = logging.getLogger(__name__)


def parse_index_list(value: str) -> list[int]:
    """Parse comma-separated indices and ranges (e.g. 3,7-10,425) into a sorted list of ints."""
    if value is None or value.strip() == "":
        return []

    indices: set[int] = set()
    parts = value.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            if not start_str or not end_str:
                raise argparse.ArgumentTypeError(f"Invalid range: {part!r}")
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid range boundaries: {part!r}")
            if start < 0 or end < 0:
                raise argparse.ArgumentTypeError("Negative indices are not supported.")
            if end < start:
                raise argparse.ArgumentTypeError(f"Range start must be <= end: {part!r}")
            indices.update(range(start, end + 1))
        else:
            try:
                idx = int(part)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid index: {part!r}")
            if idx < 0:
                raise argparse.ArgumentTypeError("Negative indices are not supported.")
            indices.add(idx)

    return sorted(indices)


def main():
    """The main function of LMbench."""
    logger.info("Starting")

    parser = argparse.ArgumentParser(description="Run LLM evaluation.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model name, e.g. 'gpt-4o', 'llama3.2:1b' or 'llama3.1-70b'.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=MODEL_MAPPING.keys(),
        help="The provider that should be used for the model, e.g. 'ollama', 'openai' or 'azure_openai'.",
    )
    parser.add_argument(
        "--model_args",
        type=str,
        default="",
        help="Arguments to pass to the specific provider class in formart key1=value1,key2=value2.",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use for evaluation")
    parser.add_argument(
        "--force-data-download",
        action="store_true",
        default=False,
        help="Forces the dataset to be downloaded regardless of it having been downloaded before.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="If set, the benchmark will be limited to the first n entries of each dataset.",
    )
    parser.add_argument(
        "-n",
        "--no-cache",
        action="store_true",
        default=False,
        help="If set, do not use cached results for datasets that have been previously evaluated.",
    )
    parser.add_argument(
        "-c",
        "--clear-cache",
        action="store_true",
        default=False,
        help="If set, clear the cache for the current llm configuration before the benchmark starts.",
    )
    parser.add_argument(
        "-t",
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use when calling the llm.",
    )
    parser.add_argument(
        "-g",
        "--skip-generation",
        action="store_true",
        default=False,
        help="Will skip the generation of results and only compute the metrics.",
    )

    parser.add_argument(
        "-m",
        "--skip-metrics",
        action="store_true",
        default=False,
        help="Will skip the computation of metrics and only compute the generation.",
    )

    parser.add_argument(
        "--no-metrics-cache",
        action="store_true",
        default=False,
        help="If set, do not use cached metrics from previously evaluated.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature to use for LLM generation.",
    )
    parser.add_argument(
        "--exclude-rows",
        type=parse_index_list,
        default=[],
        help="Comma-separated list of zero-based dataset rows or ranges to omit (e.g. 3,7-10,425).",
    )

    args = parser.parse_args()

    for lg in log_config["loggers"].keys():
        log_config["loggers"][lg]["level"] = args.loglevel
    dictConfig(log_config)

    model_class = MODEL_MAPPING[args.provider]
    llm = model_class.from_arg_string(
        args.model,
        args.model_args,
        temperature=args.temperature,
    )
    quantization = quantization_from_model(provider=args.provider, model_name=args.model)

    if args.clear_cache:
        logger.info(f"Clearing cache for model {llm.model} with configuration {llm.arg_string}")
        llm.clear_cache()

    dataset = load_dataset(args.dataset, args.force_data_download)
    # if dataset requests multiple iterations, wrap it
    n_iter = dataset.config_dict.get("n_iterations", 1)
    if n_iter > 1:
        if not args.no_cache:
            parser.error("Datasets with n_iterations > 1 require --no-cache. Please rerun with --no-cache.")
        if args.temperature is None or args.temperature <= 0:
            parser.error(
                "Datasets with n_iterations > 1 require --temperature > 0. Please rerun with --temperature <value>."
            )
        dataset = MultiIterationDataset(dataset, n_iter)
        logger.info("Multi-iteration dataset used; cache has been disabled by user.")

    # Set the task tag for the current run in MLflow. More tags can be added here as needed.
    mlflow.set_tag("task", dataset.task.value)
    mlflow.set_tag("quantization", quantization)

    er = EvaluationRunner(
        llm,
        dataset,
        limit=args.limit,
        excluded_indices=args.exclude_rows,
    )
    if args.no_cache:
        logger.info("Not using cache in this run.")
    er.run(
        use_cache=not args.no_cache,
        num_threads=args.num_threads,
        skip_generation=args.skip_generation,
        skip_metrics=args.skip_metrics,
        use_metrics_cache=not args.no_metrics_cache,
    )


if __name__ == "__main__":
    main()
