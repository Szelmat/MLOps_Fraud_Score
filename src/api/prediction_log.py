"""Log of every scored transaction -- the input of the Evidently drift report.

One JSON line per transaction: its features in the training data's column names, so the
file compares column by column with data/processed/train.csv, plus the prediction.
"""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from src.api.schemas import Prediction, Transaction

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
