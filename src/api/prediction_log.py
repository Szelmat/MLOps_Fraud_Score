"""Log of every scored transaction -- the input of the Evidently drift report.

One JSON line per transaction: its features in the training data's column names, so the
file compares column by column with data/processed/train.csv, plus the prediction.
At API shutdown the file is stored in MLflow as an artifact of a "<experiment>-serving" run.
"""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import structlog
from mlflow.exceptions import MlflowException

from src.api.schemas import Prediction, Transaction
from src.ml import config

logger = structlog.get_logger()
# Requests run in a thread pool; the lock keeps their lines from interleaving.
_lock = threading.Lock()


def log_predictions(path: Path, transactions: list[Transaction], predictions: list[Prediction]) -> None:
    logged_at = datetime.now(UTC).isoformat()
    lines = [
        json.dumps({"logged_at": logged_at} | transaction.to_feature_row() | prediction.model_dump())
        for transaction, prediction in zip(transactions, predictions, strict=True)
    ]
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")


def upload_prediction_log(path: Path) -> None:
    """Stores the log in MLflow. Failing to archive must not crash the API, so errors are only logged."""
    if not path.exists():
        return
    try:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(f"{config.MLFLOW_EXPERIMENT_NAME}-serving")
        with mlflow.start_run(run_name=f"serving {datetime.now():%m%d_%H%M}"):
            mlflow.log_artifact(str(path))
    except MlflowException:
        logger.exception("prediction_log_upload_failed")
