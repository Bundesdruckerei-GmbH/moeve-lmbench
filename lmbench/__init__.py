"""LMBench."""

import os

# mlflow 3.13 defaults to a sqlite backend that mlflow-skinny can't use (no alembic);
# pin to the file store. setdefault so an external MLFLOW_TRACKING_URI still wins.
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
