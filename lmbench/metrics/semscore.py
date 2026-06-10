"""Contains the semscore metric that uses embeddings and cosine distance to compare two texts."""

import logging
from typing import override

import numpy as np
import torch
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_delay, wait_fixed
from torch.cuda import OutOfMemoryError
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lmbench.config.config import CONFIG
from lmbench.metrics.abstract import Metric
from lmbench.task import Task

logger = logging.getLogger(__name__)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate the cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): The first vector
        vec2 (np.ndarray): The second vector

    Returns:
        float: The cosine similarity between vec1 and vec2
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def cos_similarity_to_distance(cosine_sim: float) -> float:
    """Convert a cosine similarity to a normalized distance between 0 and 1.

    Args:
        cosine_sim (float): The cosine similarity between two vectors

    Returns:
        float: The distance between the two vectors in the rage 0-1, where 0 is identical and 1 is completely different.
    """
    return (1 - cosine_sim) / 2


class SemScore(Metric):
    """SemScore metric that uses embeddings and cosine distance to compare two texts."""

    name = "semscore"
    tasks: list[Task] = [Task.SUMMARIZATION, Task.QUESTION_ANSWERING, Task.SEMANTICSIMILARITY]

    """The safety margin of the context size in number of characters"""
    CONTEXT_SIZE_SAFETY_MARGIN = 50

    @override
    def description(self) -> str:
        return "This metric measures the semantic similarity between two texts using embeddings and cosine distance."

    @override
    def __init__(self):
        super().__init__()
        self.embedding_model = AutoModel.from_pretrained(
            CONFIG.semscore.embedding_model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        self.ctx_size = CONFIG.semscore.embedding_model_ctx_size
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG.semscore.embedding_model_name, model_max_length=None)

    @retry(
        reraise=True,
        stop=stop_after_delay(5 * 3600),  # retry for up to 5 hours
        wait=wait_fixed(5),  # wait 5s between retries
        retry=retry_if_exception_type(OutOfMemoryError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _move_model_to_cuda(self):
        """
        Retry moving model to CUDA for up to 5 hours, waiting 5s between attempts,
        on torch.cuda.OutOfMemoryError.
        """
        try:
            torch.cuda.empty_cache()
            self.embedding_model.to("cuda")
        except OutOfMemoryError:
            logger.warning("SemScore: CUDA OOM when moving model to CUDA; will retry")
            torch.cuda.empty_cache()
            raise

    def ctx_size_factor(self, text: str) -> float:
        """Calculate the context size factor based on the number of tokens in the text.

        Args:
            text (str): The input text

        Returns:
            float: How much the context size of the embedding model is filled, with 1 being full, >1 being overfull and
                0 being empty.
        """
        return len(self.tokenizer.encode(text, add_special_tokens=True)) / self.ctx_size

    def get_embedding(self, input_strings: list[str]) -> list[list[float]]:
        """Generates an embedding for the given list of input strings.

        This method implements retries when a rate limit error is raised.
        This method splits the input into smaller slices in order to fit into one embedding call.

        Args:
            input_strings (list[str]): The list of texts that should be embedded.

        Returns:
            list[list[float]]: A list of embeddings for each input string in the form of a list of floats.
        """
        embeds = []
        CHUNK_SIZE = CONFIG.semscore.chunk_size
        for i in tqdm(range(0, len(input_strings), CHUNK_SIZE)):
            current_slice = input_strings[i : i + CHUNK_SIZE]
            resp: np.ndarray = self.embedding_model.encode(current_slice)
            embeds.extend(resp.tolist())
        return embeds

    @override
    def evaluate(self, model_output: list[str], labels: list[str]) -> dict[str, list[float]]:
        # pick target device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.debug(f"SemScore.evaluate: selected device {device}")

        # move model onto device, with retry on OOM
        if device == "cuda":
            logger.debug("SemScore.evaluate: attempting to move embedding_model to CUDA")
            self._move_model_to_cuda()
        elif device == "mps":
            logger.debug("SemScore.evaluate: moving embedding_model to MPS")
            self.embedding_model.to("mps")

        # Ensuring the model output and labels fit into the context size of the embedding model
        for i in range(len(model_output)):
            label_ctx_factor = self.ctx_size_factor(labels[i])
            if label_ctx_factor > 1.0:
                logger.info(f"Label {i} is too long, truncating to fit context size.")
                labels[i] = labels[i][: int(len(labels[i]) * (1 / label_ctx_factor)) - self.CONTEXT_SIZE_SAFETY_MARGIN]
            model_ctx_factor = self.ctx_size_factor(model_output[i])
            if model_ctx_factor > 1.0:
                logger.info(f"Label {i} is too long, truncating to fit context size.")
                model_output[i] = model_output[i][
                    : int(len(model_output[i]) * (1 / model_ctx_factor)) - self.CONTEXT_SIZE_SAFETY_MARGIN
                ]

        # Bug fix for runs where cached llm output is already None.
        # This bug has been fixed, but still the cache might have None values.
        model_output = [" " if mo is None or mo == "" else mo for mo in model_output]
        labels = [" " if label is None or label == "" else label for label in labels]

        # Calculate the embeddings
        logger.debug(f"Embedding the model output ({len(model_output)}).")
        model_embeddings = self.get_embedding(model_output)
        logger.debug(f"Embedding the labels ({len(model_output)}).")
        label_embeddings = self.get_embedding(labels)

        # Calculate the cosine similarity between each pair of embeddings
        cos_sims = []
        for model_embedding, label_embedding in zip(model_embeddings, label_embeddings):
            cos_sim = cosine_similarity(np.array(model_embedding), np.array(label_embedding))
            cos_sims.append(cos_sim)
        # Convert cosine similarity to distance
        cos_dist = [cos_similarity_to_distance(x) for x in cos_sims]
        try:
            return {"norm_cos_dist": cos_dist, "cos_sim": cos_sims}
        finally:
            logger.debug("SemScore.evaluate: moving embedding_model back to CPU")
            self.embedding_model.to("cpu")
            if device == "cuda":
                logger.debug("SemScore.evaluate: clearing CUDA cache")
                torch.cuda.empty_cache()
