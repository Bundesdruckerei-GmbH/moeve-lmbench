"""Dataset wrapper that repeats each example multiple times."""

from lmbench.dataset.abstract import Dataset
from lmbench.models.data_models import LLMMessage


class MultiIterationDataset(Dataset):
    """Wraps any Dataset and repeats each example n times."""

    def __init__(self, base: Dataset, n: int):
        """Initialize with base dataset and number of iterations."""
        # proxy to the base dataset
        self.base = base
        self.n = n
        self.name = base.name
        self.task = base.task
        self.filename = base.filename
        self.config_dict = base.config_dict

    def __len__(self) -> int:
        """Total number of examples (base length multiplied by iterations)."""
        return len(self.base) * self.n

    def llm_input_at_index(self, index: int, shrink_factor: float | list[float] = 1.0) -> list[LLMMessage]:
        """Retrieve LLM input messages for the given (possibly repeated) index."""
        real_idx = index % len(self.base)
        return self.base.llm_input_at_index(real_idx, shrink_factor=shrink_factor)

    @property
    def num_shrinkable(self) -> int:
        """Proxy to the wrapped dataset's shrinkable dimensions."""
        return self.base.num_shrinkable

    def item_ids(self, limit: int | None = None) -> list[str]:
        """Return item ids repeated up to the specified limit.

        Repetition is interleaved rather than blocked, matching `llm_input_at_index`'s `index % len(base)`,
        so each id legitimately appears `n` times. Callers distinguish the repeats by generation index.
        """
        ids = self.base.item_ids(limit=None)
        repeated = ids * self.n
        if limit is not None and limit > 0 and limit < len(repeated):
            return repeated[:limit]
        return repeated

    def generation_indices(self, limit: int | None = None) -> list[int | None]:
        """Return the repeat number of each row, matching `item_ids`' interleaved repetition."""
        base_length = len(self.base)
        indices: list[int | None] = [i // base_length for i in range(base_length * self.n)]
        if limit is not None and limit > 0 and limit < len(indices):
            return indices[:limit]
        return indices

    def target_values(self, limit: int | None = None) -> list[str]:
        """Return target values repeated up to the specified limit."""
        vals = self.base.target_values(limit=None)
        repeated = vals * self.n
        if limit is not None and limit > 0 and limit < len(repeated):
            return repeated[:limit]
        return repeated

    def get_column(self, name: str) -> list:
        """Get a column from the base dataset repeated n times."""
        base_column = list(self.base.get_column(name))
        return base_column * self.n
