"""Module that takes care of running the actual evaluation."""

import concurrent.futures
import json
import logging
import os.path

import mlflow
import pandas as pd
from tqdm import tqdm

from lmbench.config.config import OUTPUT_FOLDER
from lmbench.dataset.abstract import Dataset
from lmbench.dataset.dataframe import DataframeDataset
from lmbench.emissions import EmissionTracker
from lmbench.metrics import metrics_by_task
from lmbench.metrics.abstract import DatasetAwareMetric, LLMAwareMetric, Metric
from lmbench.metrics.aggregate import aggregate
from lmbench.models.abstract import LLM
from lmbench.models.data_models import LLMOutput
from lmbench.models.exceptions import ContextSizeException

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """This class represents a test case for the benchmark. It runs a given model on a given dataset
    and collects all available (or given) metrics. The result is returned and optionally written to
    a file.
    """

    def __init__(
        self,
        llm: LLM,
        dataset: Dataset,
        limit: int | None = None,
        excluded_indices: list[int] | None = None,
    ):
        """Initializes the test case with the given model, dataset and metrics.

        Args:
            llm (LLM): The model to run the test case on.
            dataset (Dataset): The dataset to run the test case on.
            limit (int, optional): The maximum number of rows of the dataset to run the task on.
                Defaults to None, which means that all rows are used.
            excluded_indices (list[int] | None): Dataset row indices that should be skipped.
        """
        self.llm = llm
        self.dataset = dataset

        self.limit = limit
        self.excluded_indices = set(excluded_indices or [])

    def run(
        self,
        metrics: list[type[Metric]] | None = None,
        use_cache: bool = True,
        num_threads: int = 1,
        skip_generation: bool = False,
        skip_metrics: bool = False,
        use_metrics_cache: bool = True,
    ):
        """
        Executes the test case and returns the result dataframe. The metrics used are provided by the
        classes in the metrics argument. If no metrics are provided, all metrics that are compatible with
        the task of the dataset are utilized.

        The resulting dataframe consists of three columns:
            - `metric`: The name of the metric (e.g., BLEU).
            - `score`: The score of the metric (e.g., precision).
            - `value`: The value of the score (e.g., 0.5).

        Args:
            metrics (list[type[Metric]], optional): The metrics to be used for the test case. If no metrics are
                provided, all metrics that are compatible with the task of the dataset are utilized. Defaults to None.
            use_cache (bool, optional): Determines if the run should use the cache. Defaults to True.
            num_threads (int, optional): Specifies the number of threads to be used for the execution. Defaults to 1.
            skip_generation (bool, optional): If set to True, the generation step is skipped. Defaults to False.
            skip_metrics (bool, optional): If set to True, the metrics computation step is skipped. Defaults to False.
            use_metrics_cache (bool, optional): Determines if the run should use the metrics cache. Defaults to True.

        Returns:
            pd.DataFrame | None: A DataFrame with the metrics results if skip_metrics is False, None otherwise.
        """
        selected_indices = self._selected_indices()
        if not selected_indices:
            logger.warning("No dataset entries selected after applying limit and exclusions.")

        indices_path = os.path.join(OUTPUT_FOLDER, "selected_indices.json")

        if not skip_generation:
            tracker = EmissionTracker(experiment_name=f"{self.llm.model}-{self.dataset.name}")
            result = self._compute_generations(selected_indices, num_threads, use_cache)
            tracker.stop(self.llm.cache_hit_counter, len(selected_indices))
            full_result = [r.original_output if isinstance(r, LLMOutput) else r for r in result]
            pd.DataFrame(full_result).to_parquet(os.path.join(OUTPUT_FOLDER, "generations.parquet"))
            with open(indices_path, "w") as fh:
                json.dump(selected_indices, fh)
            if isinstance(self.dataset, DataframeDataset):
                result_dataset = self.dataset.dataset.iloc[selected_indices].copy()
                result_dataset["model_output"] = result
                result_dataset["reasoning_output"] = [
                    r.reasoning_output if isinstance(r, LLMOutput) else "" for r in result
                ]
                result_dataset["original_output"] = full_result
                result_dataset.to_parquet(os.path.join(OUTPUT_FOLDER, "result_dataset.parquet"))
        else:
            all_results = pd.read_parquet(os.path.join(OUTPUT_FOLDER, "generations.parquet")).iloc[:, 0].to_list()
            if not os.path.exists(indices_path):
                logger.error("Missing selected_indices.json; cannot validate cached generations for current selection.")
                raise AssertionError("Cached generations were created without storing selected indices.")

            with open(indices_path) as fh:
                stored_indices: list[int] = json.load(fh)

            if stored_indices != selected_indices:
                logger.error(
                    "Cached generations were created for different row selections. Stored: %s, requested: %s",
                    stored_indices,
                    selected_indices,
                )
                raise AssertionError("Selected rows do not match cached generations. Please regenerate outputs.")

            if len(all_results) != len(selected_indices):
                logger.error(
                    "Cached generations count (%d) does not match selected rows (%d).",
                    len(all_results),
                    len(selected_indices),
                )
                raise AssertionError("Cached generations count does not match requested rows.")

            result = [LLMOutput.from_string(r) for r in all_results]

        if not skip_metrics:
            self._compute_metrics(metrics, result, use_metrics_cache, selected_indices)

    def _compute_metrics(self, metrics, result, use_metrics_cache, selected_indices: list[int]):
        # Get all applicable metrics if there are none given
        if metrics is None:
            metrics = metrics_by_task(self.dataset.task)

        # Get target values for evaluation
        target_values = self.dataset.target_values(limit=self.limit)
        target_values = [target_values[i] for i in selected_indices]

        output_path = os.path.join(OUTPUT_FOLDER, "output.csv")
        metrics_path = os.path.join(OUTPUT_FOLDER, "metrics.json")

        if use_metrics_cache:
            metrics_data = self._load_or_create_metrics(metrics_path)

            # decide if we will compute at least one new metric -> then validate that rows in output_df match current
            # values
            needs_validation = any(not self._metric_exists(metrics_data, metric_cls.name) for metric_cls in metrics)
            output_df = self._load_or_create_output_df(output_path, target_values, result, validate=needs_validation)
        else:
            metrics_data = {}
            output_df = pd.DataFrame({"target": target_values, "output": result})

        for metric in metrics:
            try:
                if issubclass(metric, LLMAwareMetric):
                    imetric = metric(model=self.llm)
                elif issubclass(metric, DatasetAwareMetric):
                    imetric = metric(dataset=self.dataset, selected_indices=selected_indices)
                else:
                    imetric = metric()
            except (FileNotFoundError, ImportError) as e:
                logger.warning(f"Skipping metric {metric.name} (missing dependency): {e}")
                continue
            if self._metric_exists(metrics_data, imetric.name):
                logger.info(f"metric {imetric.name} already exists, skipping computation.")
                self._log_existing_metrics(metrics_data, imetric.name)
            else:
                logger.debug(f"    Evaluating {imetric.name}")
                scores = imetric.evaluate(result, target_values)

                per_row_scores: dict[str, list[float] | float] = {}
                aggregated: dict[str, float] = {}

                for key, value in scores.items():
                    if isinstance(value, list) and len(value) == len(result):
                        # detailed scores for each dataset entry
                        per_row_scores[key] = value
                        output_df[f"{imetric.name}:{key}"] = value
                    elif isinstance(value, (int, float)):
                        # scalar – already aggregated by the metric itself
                        aggregated[key] = float(value)

                # Aggregate per-row lists (if any) and add them to the same dict
                if per_row_scores:
                    aggregated.update(
                        aggregate(
                            scores=per_row_scores,
                            aggregations=imetric.aggregations,
                            nan_policy=imetric.nan_policy,
                        )
                    )

                # Log & store
                for key, value in aggregated.items():
                    logger.info(f"Logging metric {imetric.name}:{key} = {value}")
                    mlflow.log_metric(key=f"{imetric.name}:{key}", value=value)
                    metrics_data[f"{imetric.name}:{key}"] = value

        self._save_metrics(metrics_path, metrics_data)
        output_df.to_csv(output_path, index=False)
        logger.info(f"Metrics saved to {metrics_path}")
        logger.info(f"Results saved to {output_path}")

    def _compute_generations(self, indices: list[int], num_threads: int, use_cache: bool):
        SHRINK_EPSILON = 1e-6

        def process_index(dataset_idx: int):
            num_cols = max(1, self.dataset.num_shrinkable)
            factors: list[float] = [1.0] * num_cols
            active = 0

            llm_input = self.dataset.llm_input_at_index(dataset_idx)
            token_factor = self.llm.check_tokens(llm_input)
            while token_factor > 1.0 and active < num_cols:
                new_factor = factors[active] / token_factor
                if new_factor < SHRINK_EPSILON:
                    factors[active] = 0.0
                    active += 1
                else:
                    factors[active] = new_factor
                logger.info(f"Shrinking with factors {[f'{f:.4f}' for f in factors]}")
                llm_input = self.dataset.llm_input_at_index(dataset_idx, shrink_factor=factors)
                token_factor = self.llm.check_tokens(llm_input)

            try:
                if use_cache:
                    llm_output = self.llm.cached_invoke(llm_input)
                else:
                    llm_output = self.llm.invoke(llm_input)
            except ContextSizeException as e:
                target_idx = min(active, num_cols - 1)
                factors[target_idx] *= e.shrink_factor
                LAST_CHANCE_SHRINKAGE = 0.01  # As this is our "last chance", let's reduce size by 1 % point more ;)
                if factors[target_idx] - LAST_CHANCE_SHRINKAGE > 0.0:
                    factors[target_idx] -= LAST_CHANCE_SHRINKAGE
                logger.info(f"Context Size Exceeded! Shrinking with factors {[f'{f:.5f}' for f in factors]}")
                llm_input = self.dataset.llm_input_at_index(dataset_idx, shrink_factor=factors)

                if use_cache:
                    llm_output = self.llm.cached_invoke(llm_input)
                else:
                    llm_output = self.llm.invoke(llm_input)

            # Check for empty strings or None and replace them with a space
            # Some metrics expect a non-empty output
            if llm_output is None or llm_output == "":
                llm_output = " "

            return llm_output

        if num_threads > 1:
            logger.info(f"Running with {num_threads} threads.")
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                result = list(tqdm(executor.map(process_index, indices), total=len(indices), desc="Running evaluation"))
        else:
            logger.info("Running with 1 thread.")
            result = [process_index(i) for i in tqdm(indices, desc="Running evaluation")]

        return result

    def _selected_indices(self) -> list[int]:
        """Return the dataset indices that should be processed after applying limit and exclusions."""
        upper_bound = len(self.dataset)
        if self.limit is not None and self.limit > 0:
            upper_bound = min(upper_bound, self.limit)

        selected = [i for i in range(upper_bound) if i not in self.excluded_indices]

        out_of_range = [i for i in self.excluded_indices if i >= upper_bound]
        if out_of_range:
            logger.warning(
                "Exclusion indices outside the evaluated slice (%s) were ignored: %s",
                upper_bound,
                out_of_range,
            )

        if self.excluded_indices:
            logger.info(
                "Skipping %d row(s); evaluating %d of %d available entries.",
                len(self.excluded_indices) - len(out_of_range),
                len(selected),
                upper_bound,
            )

        return selected

    @staticmethod
    def _load_or_create_metrics(json_file_path: str) -> dict:
        """Load existing metrics data from a JSON file."""
        if os.path.exists(json_file_path):
            logger.info(f"Loading metrics from {json_file_path}.")
            with open(json_file_path) as json_file:
                return json.load(json_file)
        else:
            logger.warning(f"Metrics file {json_file_path} does not exist.")
            return {}

    @staticmethod
    def _save_metrics(json_file_path: str, metrics_data: dict):
        with open(json_file_path, "w") as json_file:
            json.dump(metrics_data, json_file, indent=4)

    @staticmethod
    def _metric_exists(metrics_data: dict, prefix: str) -> bool:
        """Check if any key in metrics_data starts with the given prefix."""
        return any(key.startswith(prefix) for key in metrics_data.keys())

    @staticmethod
    def _log_existing_metrics(metrics_data: dict, prefix: str):
        """Logs existing metrics using mlflow if they start with the given metric name."""
        for metric_key, metric_value in metrics_data.items():
            if metric_key.startswith(prefix):
                mlflow.log_metric(metric_key, metric_value)

    @staticmethod
    def _load_or_create_output_df(output_path, target_values, result, validate: bool = False):
        if os.path.exists(output_path):
            output_df = pd.read_csv(output_path)
            if validate:
                for i in range(len(target_values)):
                    expected = str(target_values[i])
                    actual = str(output_df["target"].iat[i])
                    if expected != actual:
                        logger.error(f"Target mismatch at index {i}: expected {expected!r}, got {actual!r}")
                        raise AssertionError(
                            f"Cached {output_path} does not match current target values. "
                            "Delete the output folder or use --no-metrics-cache to recompute."
                        )
                for i in range(len(result)):
                    raw = output_df["output"].iat[i]
                    expected = str(result[i]).rstrip("\x00")
                    if not (pd.isna(raw) and result[i] == "N/A"):
                        actual = str(raw).rstrip("\x00") if not pd.isna(raw) else ""
                        if actual != expected:
                            logger.error(f"Output mismatch at index {i}: expected {expected!r}, got {raw!r}")
                            raise AssertionError(
                                f"Cached {output_path} does not match current output values. "
                                "Delete the output folder or use --no-metrics-cache to recompute."
                            )
        else:
            output_df = pd.DataFrame({"target": target_values, "output": result})

        return output_df
