"""Per-item score artifact: one row per (item, metric, judge) for a run.

Aggregate metrics alone cannot carry uncertainty - a mean with no `n` and no dispersion admits no
confidence interval downstream. This module writes the raw per-item scores an evaluation produced,
keyed by the `item_id` stamped into the dataset file, so the aggregating side can resample them.

The artifact deliberately holds no generations: in long format every generation would be duplicated
once per (metric, judge), and `generations.parquet` already stores them keyed by the same rows.

See `docs/per-item-scores-contract.md` on the dashboard side for the contract this implements.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from lmbench.config.config import CONFIG, JudgeLLMConfig
from lmbench.dataset.abstract import Dataset
from lmbench.task import Task

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lmbench.metrics.abstract import Metric
    from lmbench.models.abstract import LLM

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
"""Contract version stored in the file-level Parquet metadata. Consumers reject unknown majors."""

ARTIFACT_FILENAME = "per_item_scores.parquet"


class ItemStatus(str, Enum):
    """Why a per-item score is or is not usable.

    `score` is null for every status other than `OK`. Consumers treat unknown non-`ok` statuses as
    "not scored", so new values can be added without a schema bump.

    Attributes:
        OK: The score is valid.
        MODEL_INVALID: The model output was missing or unparseable.
        JUDGE_FAILED: The judge or evaluator errored, or returned nothing usable.
    """

    OK = "ok"
    MODEL_INVALID = "model_invalid"
    JUDGE_FAILED = "judge_failed"


@dataclass(frozen=True)
class RowIdentity:
    """Identity and resampling structure of one evaluated row.

    Attributes:
        item_id: The stamped dataset identifier, stable across models and runs.
        cluster_id: A resampling unit. Rows sharing a value are drawn or dropped together.
        cluster_id_2: A second resampling unit, crossing `cluster_id` rather than nesting inside it.
            The two are symmetric: a consumer draws both margins in the same replicate.
        stratum: A fixed design cell. Resampling happens within a stratum, never across.
        generation_index: Which repeat of the item this row is, or None if answered once.
        attributes: Grouping keys a task's aggregates need beyond the cluster, e.g. which version of a pair the
            row is. None for tasks whose aggregates are plain means.
    """

    item_id: str
    cluster_id: str | None
    cluster_id_2: str | None
    stratum: str | None
    generation_index: int | None
    attributes: dict[str, str] | None = None


ARTIFACT_SCHEMA = pa.schema(
    [
        pa.field("item_id", pa.string(), nullable=False),
        pa.field("cluster_id", pa.string()),
        pa.field("cluster_id_2", pa.string()),
        pa.field("stratum", pa.string()),
        pa.field("generation_index", pa.int32()),
        pa.field("attributes", pa.map_(pa.string(), pa.string())),
        pa.field("metric", pa.string(), nullable=False),
        pa.field("judge", pa.string()),
        pa.field("score", pa.float64()),
        pa.field("status", pa.string(), nullable=False),
    ]
)


def judge_config_for(metric: Metric) -> JudgeLLMConfig | None:
    """Return the judge LLM config backing a metric's scores, if it has one."""
    attribute = getattr(metric, "judge_llm_config", None)
    if attribute is None:
        return None
    return getattr(CONFIG, attribute, None)


def judge_identity(config: JudgeLLMConfig | None) -> str | None:
    """Build the `judge` column value for a judge config.

    The deployment qualifies the model because it decides which model version serves the request.

    Args:
        config: The judge configuration, or None for a metric with no LLM judge.

    Returns:
        A `model:deployment` identifier, or None when the metric has no judge.
    """
    if config is None:
        return None
    if config.azure_deployment:
        return f"{config.model}:{config.azure_deployment}"
    return config.model


def _column(dataset: Dataset, name: str, selected_indices: list[int]) -> list[Any]:
    """Return one dataset column restricted to the evaluated rows."""
    values = list(dataset.get_column(name))
    return [values[i] for i in selected_indices]


def row_identities(dataset: Dataset, selected_indices: list[int], limit: int | None = None) -> list[RowIdentity]:
    """Resolve item id, cluster, stratum and generation index for every evaluated row.

    Clusters and strata are structure *above* the item, and the two tasks that have them need
    different structure, so it is resolved once here and carried in columns rather than left for the
    consumer to rediscover from an id.

    Args:
        dataset: The dataset being evaluated.
        selected_indices: The dataset row indices included in this run.
        limit: The `--limit` applied to the run, passed through to the dataset accessors.

    Raises:
        MissingItemIdError: If the dataset carries no stamped item id column.

    Returns:
        One identity per evaluated row, in evaluation order.
    """
    all_ids = dataset.item_ids(limit=limit)
    all_generations = dataset.generation_indices(limit=limit)
    item_ids = [all_ids[i] for i in selected_indices]
    generation_indices = [all_generations[i] for i in selected_indices]

    clusters: list[str | None] = [None] * len(item_ids)
    clusters_2: list[str | None] = [None] * len(item_ids)
    strata: list[str | None] = [None] * len(item_ids)
    attributes: list[dict[str, str] | None] = [None] * len(item_ids)

    task = getattr(dataset, "task", None)
    if task == Task.HALLUCINATION:
        # The raw question type, not the scoring one: SIMPLE and COMPLEX are separate design cells
        # even though both route to ANSWERABLE. Only casing is normalized, so a variant spelling
        # cannot split a stratum in two.
        strata = [str(t).strip().upper() for t in _column(dataset, "question_type", selected_indices)]
    elif task == Task.VALUE_EVALUATION:
        # The repeated generations of an entry are not independent and move together.
        clusters = list(item_ids)
    elif task == Task.POLITICAL_PARTIES:
        # A grid, so it has two crossing margins and neither absorbs the other: a party's answers sit
        # in as many statement clusters as there are theses. Both are keyed by year because a cluster
        # has to sit wholly inside one stratum.
        years = _column(dataset, "year", selected_indices)
        statements = _column(dataset, "statement", selected_indices)
        parties = _column(dataset, "party", selected_indices)
        clusters = [f"{year}:{statement}" for year, statement in zip(years, statements, strict=True)]
        clusters_2 = [f"{year}:{party}" for year, party in zip(years, parties, strict=True)]
        strata = [str(year) for year in years]

    return [
        RowIdentity(
            item_id=item_id,
            cluster_id=cluster,
            cluster_id_2=cluster_2,
            stratum=stratum,
            generation_index=generation,
            attributes=attribute,
        )
        for item_id, cluster, cluster_2, stratum, generation, attribute in zip(
            item_ids, clusters, clusters_2, strata, generation_indices, attributes, strict=True
        )
    ]


def _is_missing(value: Any) -> bool:
    """True if a score value carries no number."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return not isinstance(value, (int, float))


class PerItemScores:
    """Collects per-item score rows over a run's metrics and writes the artifact."""

    def __init__(self, identities: list[RowIdentity]):
        """Initialize the collector for a fixed set of evaluated rows.

        Args:
            identities: One identity per evaluated row, in evaluation order.
        """
        self.identities = identities
        self.rows: list[tuple] = []
        self.covered_metrics: list[str] = []
        self.aggregate_only_metrics: list[str] = []
        self.nan_policies: dict[str, str] = {}

    def __len__(self) -> int:
        """Number of collected rows."""
        return len(self.rows)

    def add_metric(self, metric: Metric, scores: dict[str, Any]) -> None:
        """Add every per-item score a metric produced.

        Two sources are merged: score keys whose value is a list aligned to the evaluated rows, and
        keys a metric exposes only through `item_scores` because `evaluate` reports them aggregated.

        A metric that implements `item_statuses` declares which keys have per-item meaning: keys
        absent from that mapping are dropped, and a None entry drops a single row (the metric does
        not apply to that item, which is not the same as failing on it). A dropped key is still
        published, so it is recorded as having no per-item data rather than left out of both lists.

        Args:
            metric: The metric instance that produced the scores.
            scores: The metric's `evaluate` return value.
        """
        from lmbench.metrics.aggregate import is_metadata_sequence

        judge = judge_identity(judge_config_for(metric))
        declared_statuses = metric.item_statuses()
        extra_scores = metric.item_scores() or {}

        num_rows = len(self.identities)
        per_item: dict[str, list[Any]] = {
            key: value
            for key, value in scores.items()
            if isinstance(value, list)
            and len(value) == num_rows
            and key not in extra_scores
            # Text lists are per-row metadata a metric echoes into output.csv, not scores.
            and not is_metadata_sequence(value)
        }
        per_item.update({key: value for key, value in extra_scores.items() if len(value) == num_rows})

        emitted = False
        undeclared: list[str] = []
        for key, values in per_item.items():
            if declared_statuses is not None and key not in declared_statuses:
                undeclared.append(f"{metric.name}:{key}")
                continue
            statuses = declared_statuses.get(key) if declared_statuses else None
            name = f"{metric.name}:{key}"
            if self._add_key(name, judge, values, statuses, omit_none=key in extra_scores):
                emitted = True
                self.covered_metrics.append(name)
                self.nan_policies[name] = metric.nan_policy.value

        aggregate_only = [
            f"{metric.name}:{key}"
            for key, value in scores.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool) and key not in per_item
        ] + undeclared
        self.aggregate_only_metrics.extend(name for name in aggregate_only if name not in self.covered_metrics)

        if not emitted:
            logger.info("Metric %s produced no per-item scores; it will have no confidence interval.", metric.name)

    def _add_key(
        self,
        name: str,
        judge: str | None,
        values: list[Any],
        statuses: list[str | None] | None,
        omit_none: bool = False,
    ) -> bool:
        """Append the rows for one metric key. Returns True if at least one row was written.

        `omit_none` applies to metric-supplied scores, where None marks a row the key does not
        cover at all - unlike NaN, which marks a row it covers and failed on.
        """
        wrote = False
        for index, identity in enumerate(self.identities):
            declared = statuses[index] if statuses is not None else None
            if statuses is not None and declared is None:
                # The metric does not apply to this item; its denominator is legitimately smaller.
                continue

            value = values[index]
            if omit_none and value is None:
                continue
            if declared is not None:
                status = declared
            elif _is_missing(value):
                status = ItemStatus.JUDGE_FAILED.value if judge else ItemStatus.MODEL_INVALID.value
            else:
                status = ItemStatus.OK.value

            score = float(value) if status == ItemStatus.OK.value and not _is_missing(value) else None
            self._append(identity, name, judge, score, status)
            wrote = True

        return wrote

    def _append(
        self,
        identity: RowIdentity,
        metric: str,
        judge: str | None,
        score: float | None,
        status: str,
    ) -> None:
        self.rows.append(
            (
                identity.item_id,
                identity.cluster_id,
                identity.cluster_id_2,
                identity.stratum,
                identity.generation_index,
                list(identity.attributes.items()) if identity.attributes is not None else None,
                metric,
                judge,
                score,
                status,
            )
        )

    def to_table(self, metadata: dict[str, str]) -> pa.Table:
        """Build the Arrow table, with the run metadata attached at file level."""
        columns = list(zip(*self.rows)) if self.rows else [()] * len(ARTIFACT_SCHEMA)
        table = pa.Table.from_arrays(
            [pa.array(list(column), type=field.type) for column, field in zip(columns, ARTIFACT_SCHEMA, strict=True)],
            schema=ARTIFACT_SCHEMA,
        )
        return table.replace_schema_metadata({k: str(v) for k, v in metadata.items()})

    def write(self, path: str | Path, metadata: dict[str, str]) -> None:
        """Write the artifact to `path` as zstd-compressed Parquet."""
        pq.write_table(self.to_table(metadata), str(path), compression="zstd")
        logger.info("Wrote %d per-item score rows to %s", len(self.rows), path)


RECONCILE_TOLERANCE = 1e-12
"""Relative agreement required between re-aggregated rows and the published number.

Not exact equality: an archived `output.csv` and the `metrics.json` beside it can disagree in the
last bit or two when they were produced by different invocations of a metrics-cached re-run. The
bound is still three orders of magnitude tighter than a single wrong row out of a few thousand.
"""


@dataclass(frozen=True)
class AggregateMismatch:
    """A published aggregate the per-item rows fail to reproduce.

    Attributes:
        key: The published key, e.g. `BERTScore:f1:mean`.
        published: The number the run reported.
        recomputed: The number the rows aggregate to.
        relative_delta: Absolute difference over the published magnitude, or infinity if it is zero.
    """

    key: str
    published: float
    recomputed: float
    relative_delta: float


class ReconciliationError(RuntimeError):
    """Raised when the per-item rows do not reproduce the aggregates the run published."""

    def __init__(self, mismatches: list[AggregateMismatch]):
        """Report every aggregate the rows fail to reproduce."""
        self.mismatches = mismatches
        detail = "\n".join(
            f"  {mismatch.key}: published {mismatch.published!r}, rows give {mismatch.recomputed!r} "
            f"(relative delta {mismatch.relative_delta:.3g})"
            for mismatch in mismatches
        )
        super().__init__(
            f"{len(mismatches)} published aggregate(s) the per-item rows do not reproduce:\n{detail}\n"
            "The artifact would misattribute these numbers, so it was not written."
        )


def _relative_delta(published: float, recomputed: float) -> float:
    """Difference between two aggregates, relative to the published magnitude."""
    difference = abs(recomputed - published)
    if difference == 0:
        return 0.0
    if published == 0:
        return math.inf
    return difference / abs(published)


def reconcile_aggregates(
    collector: PerItemScores,
    published: Mapping[str, float],
    metrics: Mapping[str, Metric | type[Metric]],
    tolerance: float = RECONCILE_TOLERANCE,
) -> list[AggregateMismatch]:
    """Check that the collected rows re-aggregate to the numbers the run published.

    This is what makes the artifact trustworthy: it is a claim about where a published mean came
    from, and an unverified claim is worth nothing. Re-running each metric's own `aggregations` under
    its own `nan_policy` is the only check that catches a mis-joined row, a dropped denominator or a
    status inferred the wrong way round, because all three move the aggregate.

    A metric that publishes a bare scalar its rows average to - the stance proportions - is compared
    against the mean, since `aggregate` passes scalars through without an aggregation suffix.

    Args:
        collector: The collector holding the rows to verify.
        published: The run's aggregates, as written to `metrics.json`.
        metrics: `Metric.name` to the metric (class or instance) that produced it, read for
            `aggregations` and `nan_policy`.
        tolerance: Largest relative difference still counted as agreement.

    Returns:
        One entry per published aggregate the rows fail to reproduce, empty if everything agrees.
    """
    from lmbench.metrics.aggregate import AggregationType, aggregate

    metric_position = [field.name for field in ARTIFACT_SCHEMA].index("metric")
    score_position = [field.name for field in ARTIFACT_SCHEMA].index("score")
    status_position = [field.name for field in ARTIFACT_SCHEMA].index("status")

    scores_by_key: dict[str, list[float]] = {}
    for row in collector.rows:
        value = row[score_position] if row[status_position] == ItemStatus.OK.value else math.nan
        scores_by_key.setdefault(str(row[metric_position]), []).append(float(value))

    mismatches: list[AggregateMismatch] = []
    for key, scores in scores_by_key.items():
        metric = metrics.get(key.split(":", 1)[0])
        if metric is None:
            continue
        aggregations = metric.aggregations or list(AggregationType)
        recomputed = aggregate({key: scores}, aggregations, metric.nan_policy)

        checked = [name for name in recomputed if name in published]
        if not checked and key in published:
            # A scalar the rows average to, published without an aggregation suffix.
            recomputed = {key: aggregate({key: scores}, [AggregationType.MEAN], metric.nan_policy)[f"{key}:mean"]}
            checked = [key]

        for name in checked:
            delta = _relative_delta(published[name], recomputed[name])
            if delta > tolerance:
                mismatches.append(AggregateMismatch(name, published[name], recomputed[name], delta))

    return mismatches


def merge_cached_rows(collector: PerItemScores, path: str | Path, recomputed: set[str]) -> None:
    """Carry rows for metrics that were served from the metrics cache into a fresh artifact.

    A re-run with the metrics cache enabled recomputes only the metrics that are missing, so without
    this the artifact would silently shrink to whatever the last invocation happened to compute.

    Args:
        collector: The collector holding this invocation's freshly computed rows.
        path: The existing artifact, if any.
        recomputed: Names of the metrics (`Metric.name`) evaluated in this invocation. Their rows
            are dropped from the old artifact so a metric that stopped producing per-item scores
            does not leave stale ones behind.
    """
    if not os.path.exists(path):
        return
    table = pq.read_table(str(path))

    # An artifact written before an optional column existed simply lacks it; reading it as null is
    # what makes such a column additive rather than a version bump.
    columns = {
        field.name: table.column(field.name).to_pylist() if field.name in table.column_names else [None] * len(table)
        for field in ARTIFACT_SCHEMA
    }
    rows = list(zip(*columns.values()))
    metric_position = [field.name for field in ARTIFACT_SCHEMA].index("metric")
    kept = [row for row in rows if str(row[metric_position]).split(":", 1)[0] not in recomputed]
    if not kept:
        return

    carried = sorted({str(row[metric_position]) for row in kept})
    collector.rows = kept + collector.rows
    collector.covered_metrics.extend(name for name in carried if name not in collector.covered_metrics)

    previous_metadata = table.schema.metadata or {}
    previous_policies = json.loads(previous_metadata.get(b"nan_policies", b"{}"))
    for name in carried:
        collector.nan_policies.setdefault(name, previous_policies.get(name, ""))
    logger.info("Carried %d cached per-item rows for %d metric(s) into the artifact.", len(kept), len(carried))


def harness_version() -> str:
    """Return the git SHA the run was produced by, falling back to the CI-provided one."""
    for variable in ("CI_COMMIT_SHA", "GIT_COMMIT"):
        if sha := os.getenv(variable):
            return sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def dataset_version(dataset: Dataset) -> str:
    """Return a content hash of the dataset file the run read.

    Distinguishes a run against the old dataset file from one against the stamped file.
    """
    import hashlib

    path = dataset.dataset_folder() / dataset.filename
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def compact_indices(indices: list[int]) -> str:
    """Render an index list as collapsed ranges, e.g. `0-762,900`."""
    if not indices:
        return ""
    parts: list[str] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    parts.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(parts)


def build_run_metadata(
    llm: LLM,
    dataset: Dataset,
    identities: list[RowIdentity],
    selected_indices: list[int],
    collector: PerItemScores,
) -> dict[str, str]:
    """Assemble the file-level metadata that makes the artifact self-contained.

    Args:
        llm: The evaluated model.
        dataset: The evaluated dataset.
        identities: The evaluated rows.
        selected_indices: The dataset row indices this run covered.
        collector: The collector holding the run's rows and metric coverage.

    Returns:
        A flat string-to-string mapping, ready for Parquet key-value metadata.
    """
    from lmbench.version import __version__

    judge = CONFIG.judge_llm
    hallucination_judge = CONFIG.hallucination_judge_llm
    seed = getattr(llm, "seed", None)
    if seed is None:
        seed = getattr(llm, "_kwargs", {}).get("seed")

    metadata: dict[str, str] = {
        "schema_version": str(SCHEMA_VERSION),
        "lmbench_version": __version__,
        "harness_version": harness_version(),
        "provenance": "native",
        "model": str(getattr(llm, "model", "")),
        "model_args": str(getattr(llm, "arg_string", "") or ""),
        "provider": type(llm).__name__,
        "temperature": "" if getattr(llm, "temperature", None) is None else str(llm.temperature),
        "seed": "" if seed is None else str(seed),
        "task": str(getattr(getattr(dataset, "task", None), "value", "")),
        "dataset": str(getattr(dataset, "name", "")),
        "dataset_file": str(getattr(dataset, "filename", "")),
        "dataset_version": dataset_version(dataset),
        "n_dataset_rows": str(len(dataset)),
        "n_rows_evaluated": str(len(identities)),
        "n_items": str(len({identity.item_id for identity in identities})),
        "selected_indices": json.dumps(selected_indices),
        "judge_model": judge.model,
        "judge_version": judge.api_version,
        "judge_deployment": judge.azure_deployment,
        "covered_metrics": json.dumps(sorted(set(collector.covered_metrics))),
        "metrics_without_per_item": json.dumps(sorted(set(collector.aggregate_only_metrics))),
        "nan_policies": json.dumps(collector.nan_policies),
    }
    if hallucination_judge is not None:
        metadata["hallucination_judge_model"] = hallucination_judge.model
        metadata["hallucination_judge_version"] = hallucination_judge.api_version
        metadata["hallucination_judge_deployment"] = hallucination_judge.azure_deployment
    mlflow_run = _active_run_id()
    if mlflow_run:
        metadata["mlflow_run_id"] = mlflow_run
    return metadata


def _active_run_id() -> str:
    """Return the active MLflow run id, or an empty string if there is none."""
    import mlflow

    run = mlflow.active_run()
    return run.info.run_id if run is not None else ""


MLFLOW_TAG_LIMIT = 4900
"""MLflow rejects tag values above 5000 characters."""


def run_metadata_tags(metadata: dict[str, str]) -> dict[str, str]:
    """Render the run metadata as `per_item.*` tag keys and length-capped values.

    `selected_indices` is rendered as collapsed ranges rather than a JSON list: the full list runs to
    thousands of entries and would be rejected by MLflow's tag length limit.
    """

    def capped(value: str) -> str:
        return value[:MLFLOW_TAG_LIMIT] + "…" if len(value) > MLFLOW_TAG_LIMIT else value

    tags = {f"per_item.{key}": capped(value) for key, value in metadata.items() if key != "selected_indices"}
    tags["per_item.selected_indices"] = capped(compact_indices(json.loads(metadata.get("selected_indices", "[]"))))
    return tags


def log_run_metadata_tags(metadata: dict[str, str]) -> None:
    """Mirror the run metadata onto the active MLflow run so it is queryable without the artifact."""
    import mlflow

    for key, value in run_metadata_tags(metadata).items():
        mlflow.set_tag(key, value)
