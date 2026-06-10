"""Module for SQLite-backed cache manager with thread safety."""

import os
import threading
from pathlib import Path
from typing import Any

import dotenv
from sqlitedict import SqliteDict

dotenv.load_dotenv()


class CacheManager:
    """SQLite-backed cache manager with thread safety."""

    def __init__(self, db_path: Path, lock: threading.Lock, model_name: str = ""):
        """Initialize CacheManager with database path and threading lock."""
        self.lock = lock
        self.db = SqliteDict(
            filename=str(db_path),
            autocommit=True,
        )
        # set model_name only if running locally by checking LOCAL env variable
        if os.getenv("LOCAL", "false") == "true":
            self.model_name = model_name
        else:
            self.model_name = ""

    def get(self, key: str) -> Any | None:
        """Retrieve any picklable value by key."""
        with self.lock:
            return self.db.get(f"{key}{self.model_name}", None)

    def set(self, key: str, val: Any) -> None:
        """Store any picklable value under the given key."""
        with self.lock:
            self.db[f"{key}{self.model_name}"] = val

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self.lock:
            self.db.clear()
