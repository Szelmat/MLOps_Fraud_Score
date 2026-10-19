import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient
from src.api.main import app
from src.api.prediction_log import log_predictions, upload_prediction_log
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
    mlflow_client = MlflowClient(config.MLFLOW_TRACKING_URI)
    experiment = mlflow_client.get_experiment_by_name(f"{config.MLFLOW_EXPERIMENT_NAME}-serving")
    assert experiment is not None
    [run] = mlflow_client.search_runs([experiment.experiment_id])
    assert [artifact.path for artifact in mlflow_client.list_artifacts(run.info.run_id)] == ["predictions.jsonl"]


@pytest.mark.usefixtures("project")
def test_rejected_requests_are_not_logged():
    with TestClient(app) as client:
        app.state.model = FixedModel()
        client.post("/predict", json=EXAMPLE_TRANSACTION | {"risk_score": 5})
    assert not config.PREDICTION_LOG_PATH.exists()


@pytest.mark.usefixtures("project")
def test_upload_failure_is_logged_not_raised(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    config.PREDICTION_LOG_PATH.parent.mkdir(parents=True)
    config.PREDICTION_LOG_PATH.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", "unsupported://registry")
    upload_prediction_log(config.PREDICTION_LOG_PATH)
    assert "prediction_log_upload_failed" in capsys.readouterr().out
