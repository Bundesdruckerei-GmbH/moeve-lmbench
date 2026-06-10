import threading
from pathlib import Path

import pytest

from lmbench.models.cache_manager import CacheManager


@pytest.fixture
def db_path(tmp_path) -> Path:
    # each test gets its own file
    return tmp_path / "cache.sqlite"


@pytest.fixture
def cache(db_path: Path) -> CacheManager:
    lock = threading.Lock()
    return CacheManager(db_path, lock)


def test_get_missing_returns_none(cache: CacheManager):
    assert cache.get("no-such-key") is None


def test_set_and_get(cache: CacheManager):
    cache.set("foo", "bar")
    assert cache.get("foo") == "bar"


def test_overwrite_value(cache: CacheManager):
    cache.set("key", "first")
    cache.set("key", "second")
    assert cache.get("key") == "second"


def test_clear(cache: CacheManager):
    cache.set("a", "1")
    cache.set("b", "2")
    cache.clear()
    assert cache.get("a") is None
    assert cache.get("b") is None


def test_persistence_across_instances(db_path: Path):
    # write with one manager
    lock1 = threading.Lock()
    cm1 = CacheManager(db_path, lock1)
    cm1.set("persist", "yes")

    # read with another manager pointing at same file
    lock2 = threading.Lock()
    cm2 = CacheManager(db_path, lock2)
    assert cm2.get("persist") == "yes"


def test_thread_safety(db_path: Path):
    lock = threading.Lock()
    cm = CacheManager(db_path, lock)

    def writer(idx: int):
        cm.set(f"k{idx}", f"v{idx}")

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i in range(20):
        assert cm.get(f"k{i}") == f"v{i}"
