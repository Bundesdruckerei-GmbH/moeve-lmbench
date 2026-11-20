from lmbench.dataset.abstract import Dataset
from lmbench.dataset.multi_iteration import MultiIterationDataset
from lmbench.models.data_models import LLMMessage, LLMRole


class DummyDataset(Dataset):
    """A minimal Dataset stub for testing."""

    def __init__(self):
        self.name = "dummy"
        self.task = "task"
        self.filename = "dummy.csv"
        self.config_dict = {}
        self.config = None

    @staticmethod
    def type() -> str:
        return "dummy"

    def __len__(self) -> int:
        return 2

    def llm_input_at_index(self, index: int, shrink_factor: float = 1.0) -> list[LLMMessage]:  # noqa: ARG002
        return [LLMMessage(role=LLMRole.USER, content=f"msg{index}")]

    def target_values(self, limit: int | None = None) -> list[str]:
        vals = ["a", "b"]
        if limit is not None:
            return vals * (limit // len(vals)) + vals[: limit % len(vals)]
        return vals

    def get_column(self, _name: str) -> list:
        return []


def test_length():
    base = DummyDataset()
    mid = MultiIterationDataset(base, 3)
    assert len(mid) == 2 * 3


def test_llm_input_repeats():
    base = DummyDataset()
    mid = MultiIterationDataset(base, 3)
    for i in range(6):
        assert mid.llm_input_at_index(i) == base.llm_input_at_index(i % len(base))


def test_target_values_no_limit():
    base = DummyDataset()
    mid = MultiIterationDataset(base, 2)
    assert mid.target_values() == ["a", "b", "a", "b"]


def test_target_values_with_limit_less():
    base = DummyDataset()
    mid = MultiIterationDataset(base, 2)
    assert mid.target_values(limit=3) == ["a", "b", "a"]


def test_target_values_with_limit_more():
    base = DummyDataset()
    mid = MultiIterationDataset(base, 2)
    # limit > total, should return full repeated list
    assert mid.target_values(limit=10) == ["a", "b", "a", "b"]


def test_generation_indices_align_with_the_interleaved_repeats():
    base = DummyDataset()
    mid = MultiIterationDataset(base, 3)
    assert mid.generation_indices() == [0, 0, 1, 1, 2, 2]


def test_generation_indices_with_limit():
    base = DummyDataset()
    mid = MultiIterationDataset(base, 3)
    assert mid.generation_indices(limit=3) == [0, 0, 1]


def test_base_dataset_reports_no_generation_index():
    assert DummyDataset().generation_indices() == [None, None]
