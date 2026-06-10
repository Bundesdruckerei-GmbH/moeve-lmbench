"""The metric module that contains the metrics for every type of task."""

from lmbench.metrics.abstract import Metric
from lmbench.metrics.hfevaluate import BLEU, ROUGE, BERTScore
from lmbench.metrics.is_german import IsGerman
from lmbench.metrics.match import Match
from lmbench.metrics.ragas_metrics import RagasComparisonMetrics, RagasQAMetrics, RagasTopicExtractionMetrics
from lmbench.metrics.semscore import SemScore
from lmbench.metrics.sklearnevaluate import ClassificationReport, MultiLabelSetEvaluation
from lmbench.metrics.sustainability import SustainabilityMetrics
from lmbench.metrics.topic_extraction import SemScoreTopicExtraction, TopicExtraction
from lmbench.metrics.values import ValuesMetric, ValuesMetricBERT
from lmbench.metrics.wahl_o_mat import PoliticalPartiesEvaluation, WahlOMatEvaluation
from lmbench.task import Task

METRICS: list[type[Metric]] = [
    BERTScore,
    BLEU,
    Match,
    ROUGE,
    SemScore,
    RagasQAMetrics,
    RagasComparisonMetrics,
    TopicExtraction,
    RagasTopicExtractionMetrics,
    SemScoreTopicExtraction,
    ClassificationReport,
    WahlOMatEvaluation,
    MultiLabelSetEvaluation,
    IsGerman,
    SustainabilityMetrics,
    ValuesMetric,
    ValuesMetricBERT,
    PoliticalPartiesEvaluation,
]
METRIC_MAPPING = {m.name: m for m in METRICS}


def metrics_by_task(task: Task) -> list[type[Metric]]:
    """Returns a list of metric classes that can be used for the given task. This function uses the
    METRICS list to determine which metrics are available for the given task.

    Args:
        task (Task): A task for which metrics should be returned.

    Returns:
        list[Metric]: A list of metric classes that can be used for the given task.
    """
    return [m for m in METRICS if task in m.tasks]
