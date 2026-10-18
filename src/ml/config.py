"""Shared configuration and helpers for the fraud pipeline.

Single source of truth for paths and for reading ``params.yaml``.
No magic constants scattered across the stage modules.
"""

import os
from pathlib import Path
from typing import Any

import yaml

# --- Paths -----------------------------------------------------------------
# config.py -> ml -> src -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
METRICS_DIR: Path = PROJECT_ROOT / "metrics"

RAW_CSV: Path = RAW_DIR / "fraud_5000.csv"
TRAIN_CSV: Path = PROCESSED_DIR / "train.csv"
VALID_CSV: Path = PROCESSED_DIR / "valid.csv"
METRICS_PATH: Path = METRICS_DIR / "metrics.json"
# Every transaction the API scores (JSON Lines), the input of drift reports.
PREDICTION_LOG_PATH: Path = PROJECT_ROOT / "logs" / "predictions.jsonl"

PARAMS_PATH: Path = PROJECT_ROOT / "params.yaml"
DVC_LOCK_PATH: Path = PROJECT_ROOT / "dvc.lock"

# --- MLflow ----------------------------------------------------------------
# The MLflow server of docker-compose.yaml, reached from the host.
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5500")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-score")
MLFLOW_REGISTERED_MODEL: str = os.getenv("MLFLOW_REGISTERED_MODEL", "fraud-score")
# The registry alias of the version the API serves; evaluate.py moves it.
CHAMPION_ALIAS: str = "champion"
# train writes the run_id here; evaluate reads it to resume the same run
MLFLOW_RUN_ID_PATH: Path = MODELS_DIR / "mlflow_run_id"


# --- Params ----------------------------------------------------------------
def load_params(path: Path | None = None) -> dict[str, Any]:
    """Read ``params.yaml``. Every hyperparameter lives there, never in code."""
    path = path or PARAMS_PATH
    with path.open("r", encoding="utf-8") as fh:
        params: dict[str, Any] = yaml.safe_load(fh)
    return params
