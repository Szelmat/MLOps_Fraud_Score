from pathlib import Path
from typing import Any, get_args

import numpy as np
import pandas as pd
import pytest
import yaml

from src.api import schemas
from src.ml import config

PARAMS: dict[str, Any] = {
    "seed": 42,
    "prepare": {
        "test_size": 0.25,
        "target": "Fraud_Label",
        "drop_columns": ["Transaction_ID", "User_ID", "Timestamp"],
        "stratify": True,
    },
    "train": {
        "model": "random_forest",
        "n_estimators": 20,
        "max_depth": 4,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
    },
    "evaluate": {"threshold": 0.5, "min_f1": 0.3},
}


def make_transactions(n: int, seed: int = 0) -> pd.DataFrame:
    """Synthetic rows in the raw dataset's columns and formats, so tests never need the DVC data.

    Fraud depends on Risk_Score and Failed_Transaction_Count_7d plus noise: learnable, not perfect.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 365 * 24 * 60, n), unit="min")
    risk = rng.uniform(0, 1, n)
    failed = rng.integers(0, 5, n)
    fraud = (6 * (risk - 0.6) + 0.8 * (failed - 2) + rng.normal(0, 1, n) > 0).astype(int)

    def choice(literal: Any) -> Any:
        return rng.choice(list(get_args(literal)), n)

    return pd.DataFrame(
        {
            "Transaction_ID": [f"TXN_{i}" for i in range(n)],
            "User_ID": [f"USER_{i}" for i in rng.integers(0, n // 4, n)],
            "Transaction_Amount": rng.uniform(1, 1000, n).round(2),
            "Transaction_Type": choice(schemas.TransactionType),
            "Timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "Account_Balance": rng.uniform(0, 100_000, n).round(2),
            "Device_Type": choice(schemas.DeviceType),
            "Location": choice(schemas.Location),
            "Merchant_Category": choice(schemas.MerchantCategory),
            "IP_Address_Flag": rng.integers(0, 2, n),
            "Previous_Fraudulent_Activity": rng.integers(0, 2, n),
            "Daily_Transaction_Count": rng.integers(0, 15, n),
            "Avg_Transaction_Amount_7d": rng.uniform(1, 500, n).round(2),
            "Failed_Transaction_Count_7d": failed,
            "Card_Type": choice(schemas.CardType),
            "Card_Age": rng.integers(1, 240, n),
            "Transaction_Distance": rng.uniform(0, 5000, n).round(2),
            "Authentication_Method": choice(schemas.AuthenticationMethod),
            "Risk_Score": risk.round(4),
            "Is_Weekend": (timestamps.dayofweek >= 5).astype(int),
            "Fraud_Label": fraud,
        }
    )


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points every path in `config` at a temporary project with synthetic raw data and params.yaml,
    so nothing touches the repository.
    """
    paths = {
        "RAW_CSV": "data/raw/fraud_5000.csv",
        "PROCESSED_DIR": "data/processed",
        "TRAIN_CSV": "data/processed/train.csv",
        "VALID_CSV": "data/processed/valid.csv",
        "METRICS_PATH": "metrics/metrics.json",
        "PARAMS_PATH": "params.yaml",
        "DVC_LOCK_PATH": "dvc.lock",
        "MLFLOW_RUN_ID_PATH": "models/mlflow_run_id",
        "PREDICTION_LOG_PATH": "logs/predictions.jsonl",
    }
    for name, relative in paths.items():
        monkeypatch.setattr(config, name, tmp_path / relative)
    config.RAW_CSV.parent.mkdir(parents=True)
    make_transactions(400).to_csv(config.RAW_CSV, index=False)
    config.PARAMS_PATH.write_text(yaml.safe_dump(PARAMS), encoding="utf-8")
    return tmp_path
