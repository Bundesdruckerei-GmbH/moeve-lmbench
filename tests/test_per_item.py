from math import nan

import pyarrow.parquet as pq
import pytest

from lmbench.metrics.abstract import Metric
from lmbench.metrics.aggregate import NanPolicy
from lmbench.per_item import (
    ItemStatus,
    PerItemScores,
    ReconciliationError,
    RowIdentity,
    compact_indices,
    judge_identity,
    merge_cached_rows,
    reconcile_aggregates,
    row_identities,
)
from lmbench.task import Task


class FakeDataset:
    """Minimal stand-in exposing only what row identity resolution reads."""

    def __init__(self, task: Task, ids: list[str], columns: dict | None = None, generations=None):
        self.task = task
        self.name = "fake"
        self.filename = "fake.csv"
        self.ids = ids
        self.columns = columns or {}
        self.generations = generations if generations is not None else [None] * len(ids)

    def __len__(self) -> int:
        return len(self.ids)

    def item_ids(self, limit=None) -> list[str]:
        return self.ids if limit is None else self.ids[:limit]

    def generation_indices(self, limit=None) -> list:
        return self.generations if limit is None else self.generations[:limit]

    def get_column(self, name: str) -> list:
        return self.columns[name]


class FakeMetric(Metric):
    name = "Fake"
    tasks = [Task.SUMMARIZATION]

    def __init__(self, scores=None, statuses=None, judge=None, nan_policy=NanPolicy.ZERO):
        self._scores = scores
        self._statuses = statuses
        self.judge_llm_config = judge
        self.nan_policy = nan_policy

    def description(self) -> str:
        return "fake"

    def item_scores(self):
        return self._scores

    def item_statuses(self):
        return self._statuses


def _identities(count: int) -> list[RowIdentity]:
    return [
        RowIdentity(item_id=f"ds:{i}", cluster_id=None, cluster_id_2=None, stratum=None, generation_index=None)
        for i in range(count)
    ]


def _rows_as_dicts(collector: PerItemScores) -> list[dict]:
    fields = [
        "item_id",
        "cluster_id",
        "cluster_id_2",
        "stratum",
        "generation_index",
        "attributes",
        "metric",
        "judge",
        "score",
        "status",
    ]
    return [dict(zip(fields, row, strict=True)) for row in collector.rows]


def test_row_identities_without_structure_leaves_cluster_and_stratum_null():
    dataset = FakeDataset(Task.SUMMARIZATION, ["s:0", "s:1"])
    identities = row_identities(dataset, [0, 1])
    assert [i.item_id for i in identities] == ["s:0", "s:1"]
    assert all(
        i.cluster_id is None and i.cluster_id_2 is None and i.stratum is None and i.generation_index is None
        for i in identities
    )


def test_row_identities_hallucination_strata_come_from_question_type():
    dataset = FakeDataset(
        Task.HALLUCINATION,
        ["h:0", "h:1", "h:2"],
        columns={"question_type": ["SIMPLE", "UNANSWERABLE", " complex "]},
    )
    identities = row_identities(dataset, [0, 2])
    # SIMPLE and COMPLEX stay separate strata even though both score as ANSWERABLE, and casing
    # differences do not split one.
    assert [i.stratum for i in identities] == ["SIMPLE", "COMPLEX"]
    assert all(i.cluster_id is None and i.cluster_id_2 is None for i in identities)


def test_row_identities_political_parties_cluster_by_year_and_statement():
    dataset = FakeDataset(
        Task.POLITICAL_PARTIES,
        ["w:0", "w:1", "w:2"],
        columns={"year": [2021, 2021, 2025], "statement": ["A", "A", "A"], "party": ["X", "Y", "X"]},
    )
    identities = row_identities(dataset, [0, 1, 2])
    # The same statement text in different years is a different cluster.
    assert [i.cluster_id for i in identities] == ["2021:A", "2021:A", "2025:A"]
    assert [i.stratum for i in identities] == ["2021", "2021", "2025"]


def test_row_identities_political_parties_second_margin_crosses_the_first():
    dataset = FakeDataset(
        Task.POLITICAL_PARTIES,
        ["w:0", "w:1", "w:2", "w:3"],
        columns={
            "year": [2021, 2021, 2021, 2025],
            "statement": ["A", "B", "A", "A"],
            "party": ["X", "X", "Y", "X"],
        },
    )
    identities = row_identities(dataset, [0, 1, 2, 3])
    assert [i.cluster_id_2 for i in identities] == ["2021:X", "2021:X", "2021:Y", "2025:X"]
    # Crossing rather than nesting: one party spans several statement clusters, and one statement
    # cluster spans several parties.
    assert [i.cluster_id for i in identities] == ["2021:A", "2021:B", "2021:A", "2025:A"]
    # A party standing in two elections is two clusters, so neither margin straddles a stratum.
    by_cluster_2: dict[str, set[str | None]] = {}
    for identity in identities:
        by_cluster_2.setdefault(str(identity.cluster_id_2), set()).add(identity.stratum)
    assert all(len(strata) == 1 for strata in by_cluster_2.values())


def test_row_identities_value_evaluation_clusters_generations_of_one_item():
    dataset = FakeDataset(
        Task.VALUE_EVALUATION,
        ["v:0", "v:1", "v:0", "v:1"],
        generations=[0, 0, 1, 1],
    )
    identities = row_identities(dataset, [0, 1, 2, 3])
    assert [i.cluster_id for i in identities] == ["v:0", "v:1", "v:0", "v:1"]
    assert [i.generation_index for i in identities] == [0, 0, 1, 1]
    assert all(i.stratum is None and i.cluster_id_2 is None for i in identities)


def test_text_lists_are_metadata_not_scores():
    collector = PerItemScores(_identities(2))
    collector.add_metric(FakeMetric(), {"f1": [1.0, 0.0], "attack_type": ["None", "plain"]})
    rows = _rows_as_dicts(collector)
    assert {r["metric"] for r in rows} == {"Fake:f1"}
    assert {r["status"] for r in rows} == {ItemStatus.OK.value}


def test_merge_cached_rows_reads_an_artifact_written_without_attributes(tmp_path):
    path = tmp_path / "per_item_scores.parquet"
    previous = PerItemScores(_identities(1))
    previous.add_metric(FakeMetric(), {"f1": [1.0]})
    pq.write_table(previous.to_table({}).drop_columns(["attributes"]), str(path))

    class OtherMetric(FakeMetric):
        name = "Other"

    fresh = PerItemScores(_identities(1))
    fresh.add_metric(OtherMetric(), {"score": [0.0]})
    merge_cached_rows(fresh, path, recomputed={"Other"})

    assert all(row["attributes"] is None for row in _rows_as_dicts(fresh))


def test_per_row_scores_become_one_row_each():
    collector = PerItemScores(_identities(3))
    collector.add_metric(FakeMetric(), {"f1": [0.5, 1.0, 0.0]})
    rows = _rows_as_dicts(collector)
    assert [r["score"] for r in rows] == [0.5, 1.0, 0.0]
    assert {r["metric"] for r in rows} == {"Fake:f1"}
    assert {r["status"] for r in rows} == {ItemStatus.OK.value}
    assert collector.covered_metrics == ["Fake:f1"]


def test_missing_score_without_a_judge_is_model_invalid():
    collector = PerItemScores(_identities(2))
    collector.add_metric(FakeMetric(), {"f1": [nan, 1.0]})
    rows = _rows_as_dicts(collector)
    assert rows[0]["status"] == ItemStatus.MODEL_INVALID.value
    assert rows[0]["score"] is None
    assert rows[1]["status"] == ItemStatus.OK.value


def test_missing_score_with_a_judge_is_judge_failed():
    collector = PerItemScores(_identities(1))
    collector.add_metric(FakeMetric(judge="judge_llm"), {"faithfulness": [nan]})
    assert _rows_as_dicts(collector)[0]["status"] == ItemStatus.JUDGE_FAILED.value


def test_declared_none_status_drops_the_row_entirely():
    collector = PerItemScores(_identities(3))
    collector.add_metric(
        FakeMetric(statuses={"conflict_handling": [None, ItemStatus.OK.value, ItemStatus.JUDGE_FAILED.value]}),
        {"conflict_handling": [nan, 1.0, nan]},
    )
    rows = _rows_as_dicts(collector)
    # Not applicable to item 0, so it is absent rather than counted as a failure.
    assert [r["item_id"] for r in rows] == ["ds:1", "ds:2"]
    assert [r["status"] for r in rows] == [ItemStatus.OK.value, ItemStatus.JUDGE_FAILED.value]
    assert [r["score"] for r in rows] == [1.0, None]


def test_keys_absent_from_declared_statuses_are_not_emitted():
    collector = PerItemScores(_identities(2))
    collector.add_metric(
        FakeMetric(statuses={"kept": [ItemStatus.OK.value] * 2}),
        {"kept": [1.0, 1.0], "dropped": [0.0, 0.0]},
    )
    assert {r["metric"] for r in _rows_as_dicts(collector)} == {"Fake:kept"}
    # The dropped key is still published, so it has to show up as having no per-item data. A key in
    # neither list is invisible to a consumer deciding what gets a confidence interval.
    assert collector.covered_metrics == ["Fake:kept"]
    assert collector.aggregate_only_metrics == ["Fake:dropped"]


def test_metric_supplied_item_scores_cover_an_aggregate_only_key():
    collector = PerItemScores(_identities(4))
    collector.add_metric(
        FakeMetric(scores={"stance_proportion_a": [1.0, None, 0.5, None]}),
        {"stance_proportion_a": 0.75},
    )
    rows = _rows_as_dicts(collector)
    assert [r["item_id"] for r in rows] == ["ds:0", "ds:2"]
    assert [r["score"] for r in rows] == [1.0, 0.5]
    assert collector.covered_metrics == ["Fake:stance_proportion_a"]
    assert collector.aggregate_only_metrics == []


def test_scalar_only_metric_is_recorded_as_having_no_per_item_data():
    collector = PerItemScores(_identities(2))
    collector.add_metric(FakeMetric(), {"bleu": 0.42})
    assert collector.rows == []
    assert collector.aggregate_only_metrics == ["Fake:bleu"]


def test_write_roundtrips_rows_and_file_level_metadata(tmp_path):
    collector = PerItemScores(_identities(2))
    collector.add_metric(FakeMetric(judge="judge_llm"), {"f1": [0.25, nan]})
    path = tmp_path / "per_item_scores.parquet"
    collector.write(path, {"schema_version": "1", "dataset": "fake"})

    table = pq.read_table(path)
    assert table.num_rows == 2
    assert table.column("score").to_pylist() == [0.25, None]
    assert table.schema.metadata[b"schema_version"] == b"1"
    assert table.schema.metadata[b"dataset"] == b"fake"


def test_write_handles_a_run_with_no_per_item_rows(tmp_path):
    path = tmp_path / "empty.parquet"
    PerItemScores([]).write(path, {"schema_version": "1"})
    assert pq.read_table(path).num_rows == 0


def test_merge_cached_rows_keeps_untouched_metrics_and_drops_recomputed_ones(tmp_path):
    path = tmp_path / "per_item_scores.parquet"
    previous = PerItemScores(_identities(1))
    previous.add_metric(FakeMetric(), {"f1": [1.0]})
    previous.covered_metrics = ["Fake:f1"]
    previous.write(path, {"nan_policies": '{"Fake:f1": "zero"}'})

    class OtherMetric(FakeMetric):
        name = "Other"

    fresh = PerItemScores(_identities(1))
    fresh.add_metric(OtherMetric(), {"score": [0.0]})
    merge_cached_rows(fresh, path, recomputed={"Other"})

    assert {r["metric"] for r in _rows_as_dicts(fresh)} == {"Fake:f1", "Other:score"}
    assert fresh.nan_policies["Fake:f1"] == "zero"


def test_merge_cached_rows_replaces_rows_of_recomputed_metrics(tmp_path):
    path = tmp_path / "per_item_scores.parquet"
    previous = PerItemScores(_identities(1))
    previous.add_metric(FakeMetric(), {"f1": [1.0]})
    previous.write(path, {})

    fresh = PerItemScores(_identities(1))
    fresh.add_metric(FakeMetric(), {"f1": [0.0]})
    merge_cached_rows(fresh, path, recomputed={"Fake"})

    rows = _rows_as_dicts(fresh)
    assert len(rows) == 1
    assert rows[0]["score"] == 0.0


def test_merge_cached_rows_reads_an_artifact_written_without_the_second_cluster_column(tmp_path):
    path = tmp_path / "per_item_scores.parquet"
    previous = PerItemScores(_identities(1))
    previous.add_metric(FakeMetric(), {"f1": [1.0]})
    table = previous.to_table({}).drop_columns(["cluster_id_2"])
    pq.write_table(table, str(path), compression="zstd")

    class OtherMetric(FakeMetric):
        name = "Other"

    fresh = PerItemScores(_identities(1))
    fresh.add_metric(OtherMetric(), {"score": [0.0]})
    merge_cached_rows(fresh, path, recomputed={"Other"})

    rows = _rows_as_dicts(fresh)
    assert {r["metric"] for r in rows} == {"Fake:f1", "Other:score"}
    assert all(row["cluster_id_2"] is None for row in rows)
    assert [row["item_id"] for row in rows] == ["ds:0", "ds:0"]


def _collector(scores: list[float]) -> PerItemScores:
    collector = PerItemScores(_identities(len(scores)))
    collector.add_metric(FakeMetric(), {"score": scores})
    return collector


def test_rows_that_reproduce_the_published_aggregates_reconcile():
    published = {"Fake:score:mean": 0.5, "Fake:score:median": 0.5}
    assert reconcile_aggregates(_collector([1.0, 0.0]), published, {"Fake": FakeMetric()}) == []


def test_an_aggregate_the_rows_do_not_reproduce_is_reported():
    (mismatch,) = reconcile_aggregates(_collector([1.0, 0.0]), {"Fake:score:mean": 0.9}, {"Fake": FakeMetric()})
    assert (mismatch.key, mismatch.published, mismatch.recomputed) == ("Fake:score:mean", 0.9, 0.5)
    assert mismatch.relative_delta == pytest.approx(0.4 / 0.9)


def test_a_metric_that_did_not_run_is_not_reconciled_against():
    assert reconcile_aggregates(_collector([1.0]), {"Other:score:mean": 0.2}, {"Fake": FakeMetric()}) == []


def test_a_judge_failure_counts_as_nan_under_the_metrics_own_policy():
    # ZERO scores the failed row as 0, so two rows one of which failed average to 0.5, not 1.0.
    collector = PerItemScores(_identities(2))
    collector.add_metric(FakeMetric(judge="judge_llm"), {"score": [1.0, None]})
    assert reconcile_aggregates(collector, {"Fake:score:mean": 0.5}, {"Fake": FakeMetric()}) == []


def test_omit_metric_with_mixed_rows_reconciles():
    # The disposition invariant: under OMIT a mix of real / explicit-0.0 (model-broken) /
    # NaN (judge-failed) rows must re-aggregate from the parquet back to the published number.
    # The NaN row drops; the real and the 0.0 rows average to 0.4.
    collector = PerItemScores(_identities(3))
    collector.add_metric(
        FakeMetric(judge="judge_llm", nan_policy=NanPolicy.OMIT),
        {"score": [0.8, 0.0, None]},
    )
    published = {"Fake:score:mean": 0.4, "Fake:score:median": 0.4}
    assert reconcile_aggregates(collector, published, {"Fake": FakeMetric(nan_policy=NanPolicy.OMIT)}) == []


def test_the_reconciliation_error_names_every_aggregate_it_refused_over():
    mismatches = reconcile_aggregates(_collector([1.0, 0.0]), {"Fake:score:mean": 0.9}, {"Fake": FakeMetric()})
    error = ReconciliationError(mismatches)
    assert error.mismatches == mismatches
    assert "Fake:score:mean" in str(error)
    assert "not written" in str(error)


@pytest.mark.parametrize(
    ("indices", "expected"),
    [([], ""), ([3], "3"), ([0, 1, 2, 5], "0-2,5"), ([1, 3, 5], "1,3,5")],
)
def test_compact_indices(indices, expected):
    assert compact_indices(indices) == expected


def test_judge_identity_qualifies_the_model_with_the_deployment():
    class Config:
        model = "gpt-4o"
        azure_deployment = "gpt-4o-dz"

    assert judge_identity(Config()) == "gpt-4o:gpt-4o-dz"

    class NoDeployment:
        model = "gpt-4o"
        azure_deployment = ""

    assert judge_identity(NoDeployment()) == "gpt-4o"
    assert judge_identity(None) is None
