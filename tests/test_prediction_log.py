import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.prediction_log import log_predictions
from src.api.schemas import EXAMPLE_TRANSACTION, Prediction, Transaction
from src.ml import config


class FixedModel:
    def predict_proba(self, transactions: list[Transaction]) -> list[float]:
        return [0.8] * len(transactions)


def read_log() -> list[dict[str, object]]:
    return [json.loads(line) for line in config.PREDICTION_LOG_PATH.read_text(encoding="utf-8").splitlines()]


def test_log_predictions_appends_one_json_line_per_transaction(tmp_path: Path):
    path = tmp_path / "logs" / "predictions.jsonl"
    transaction = Transaction.model_validate(EXAMPLE_TRANSACTION)
    prediction = Prediction(fraud_probability=0.8, is_fraud=True, threshold=0.5)
    log_predictions(path, [transaction, transaction], [prediction, prediction])
    log_predictions(path, [transaction], [prediction])

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3
    assert lines[0]["Transaction_Amount"] == 39.85
    assert lines[0]["IP_Address_Flag"] == 1
    assert lines[0]["fraud_probability"] == 0.8
    assert "logged_at" in lines[0]


@pytest.mark.usefixtures("project")
def test_scored_transactions_are_logged_with_features():
    with TestClient(app) as client:
        app.state.model = FixedModel()
        client.post("/predict", json=EXAMPLE_TRANSACTION)
        client.post("/predict/batch", json={"transactions": [EXAMPLE_TRANSACTION] * 2})

    lines = read_log()
    assert len(lines) == 3
    assert lines[0]["Transaction_Amount"] == 39.85
    assert lines[0]["is_fraud"] is True


@pytest.mark.usefixtures("project")
def test_rejected_requests_are_not_logged():
    with TestClient(app) as client:
        app.state.model = FixedModel()
        client.post("/predict", json=EXAMPLE_TRANSACTION | {"risk_score": 5})
    assert not config.PREDICTION_LOG_PATH.exists()
