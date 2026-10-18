from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.model import MlflowFraudModel, load_champion_model
from src.api.schemas import EXAMPLE_TRANSACTION, Transaction
from src.ml import config, evaluate, prepare, train


class RecordingPipeline:
    """Stands in for the sklearn Pipeline: remembers its input, returns fixed probabilities."""

    def __init__(self) -> None:
        self.seen = pd.DataFrame()

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        self.seen = rows
        return np.array([[0.25, 0.75]] * len(rows))


def test_model_sends_raw_feature_rows_to_the_pipeline():
    pipeline = RecordingPipeline()
    transaction = Transaction.model_validate(EXAMPLE_TRANSACTION)
    assert MlflowFraudModel(pipeline).predict_proba([transaction, transaction]) == [0.75, 0.75]
    assert list(pipeline.seen.columns) == list(transaction.to_feature_row())


def test_load_champion_model(project: Path):
    assert load_champion_model() is None
    prepare.main()
    train.main()
    evaluate.main()
    model = load_champion_model()
    assert model is not None
    [probability] = model.predict_proba([Transaction.model_validate(EXAMPLE_TRANSACTION)])
    assert 0 <= probability <= 1


def test_startup_serves_the_champion(project: Path):
    prepare.main()
    train.main()
    evaluate.main()
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        response = client.post("/predict", json=EXAMPLE_TRANSACTION)
    assert response.status_code == 200
    assert 0 <= response.json()["fraud_probability"] <= 1


def test_api_starts_without_a_model_when_mlflow_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", "unsupported://registry")
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is False
        assert client.post("/predict", json=EXAMPLE_TRANSACTION).status_code == 503
