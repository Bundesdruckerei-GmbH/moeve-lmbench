"""A metric class that evaluates the proportion of German sentences in the output text."""

from pathlib import Path
from typing import override

import fasttext
from tqdm.auto import tqdm

from lmbench.config.config import RESOURCES_FOLDER
from lmbench.metrics import Metric
from lmbench.metrics.aggregate import AggregationType
from lmbench.task import Task


class IsGerman(Metric):
    """
    A metric class that evaluates the proportion of German sentences in the output text.

    This class uses a language detector to check if the given model outputs are in German.

    Attributes:
        name (str): The name of the metric.
        tasks (list): The list of tasks this metric is applicable to, such as summarization.
        aggregations (list): The types of aggregations supported by this metric, e.g., mean.
    """

    name = "IsGerman"
    tasks = [Task.SUMMARIZATION]
    aggregations = [
        AggregationType.MEAN,
    ]

    def __init__(
        self,
        threshold: float = 0.97,
    ) -> None:
        """
        Initialize the IsGerman metric.

        Ensures the FastText language identification model is downloaded if not present,
        and loads it for computing language probabilities.

        Args:
            model_path (str): Path to the FastText model binary.
            threshold (float): Probability threshold for classifying text as German.
        """
        super().__init__()
        self.model_path = Path(RESOURCES_FOLDER) / "fasttext/lid.176.bin"
        self.threshold = threshold
        # ensure parent directory exists
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        # download model if not present locally
        if not self.model_path.is_file():
            try:
                from lmbench.azure_storage import download_blob  # pyright: ignore[reportMissingImports]

                download_blob("lmbench_resources/fasttext/lid.176.bin", self.model_path)
            except ImportError:
                raise FileNotFoundError(
                    f"FastText model not found at '{self.model_path}'. "
                    f"Please download 'lid.176.bin' from https://fasttext.cc/docs/en/language-identification.html "
                    f"and place it at '{self.model_path}'."
                ) from None
        self.ft_model = fasttext.load_model(str(self.model_path))

    @override
    def description(self) -> str:
        return "Determines if the output text is German based on a threshold"

    @override
    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, list]:
        results = []
        probs = []
        for single_output in tqdm(model_output, desc=f"Evaluating {self.name}"):
            prob = self._german_prob(single_output)
            results.append(int(prob >= self.threshold))
            probs.append(prob)

        return {"is_german": results, "german_prob": probs}

    def _german_prob(self, text: str) -> float:
        labels_list, probs_list = self.ft_model.predict([text.replace("\n", " ")], k=5)
        labels = labels_list[0]  # type: ignore[reportGeneralTypeIssues]
        probs = probs_list[0]
        return float(probs[labels.index("__label__de")]) if "__label__de" in labels else 0.0
