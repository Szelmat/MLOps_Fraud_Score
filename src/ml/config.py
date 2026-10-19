"""Shared configuration and helpers for the fraud pipeline.

Single source of truth for paths and for reading ``params.yaml``.
No magic constants scattered across the stage modules.
"""

import json
import logging
import os
import subprocess
from collections.abc import Mapping
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
# The server hands out presigned MinIO links (http://minio:9000/...) for downloads, which only
# resolve inside docker compose; download through the MLflow server instead.
os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")


# --- Params ----------------------------------------------------------------
def load_params(path: Path | None = None) -> dict[str, Any]:
    """Read ``params.yaml``. Every hyperparameter lives there, never in code."""
    path = path or PARAMS_PATH
    with path.open("r", encoding="utf-8") as fh:
        params: dict[str, Any] = yaml.safe_load(fh)
    return params


# --- Logging ---------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Structured-ish stdlib logging; never ``print()`` in pipeline code."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def flatten(params: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """`{"train": {"seed": 1}}` -> `{"train.seed": 1}` -- the shape MLflow wants."""
    flat: dict[str, Any] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


# --- Small IO helpers ------------------------------------------------------
def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as file_handler:
        json.dump(payload, file_handler, indent=2, sort_keys=True)
        file_handler.write("\n")


# --- Lineage tags -----------------------------------------------------------
def git_sha() -> str:
    """Current commit, with a '-dirty' suffix if the tree has uncommitted changes.
    A '-dirty' run is NOT reproducible.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=PROJECT_ROOT, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # metadata must never break training
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def dvc_data_hash(dep_suffix: str = "fraud_5000.csv") -> str:
    """Content hash of the raw dataset, taken from dvc.lock."""
    if not DVC_LOCK_PATH.exists():
        return "unknown"
    lock = yaml.safe_load(DVC_LOCK_PATH.read_text(encoding="utf-8")) or {}
    for dep in lock.get("stages", {}).get("prepare", {}).get("deps", []):
        if dep["path"].endswith(dep_suffix):
            return str(dep.get("md5", "unknown"))
    return "unknown"


def lineage_tags() -> dict[str, str]:
    """The tags that connect a run to the code (Git) and the data (DVC)."""
    return {"git_sha": git_sha(), "dvc_data_hash": dvc_data_hash()}
