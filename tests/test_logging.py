import json
from collections.abc import Iterator
from typing import Any

import pytest
import structlog
from fastapi.testclient import TestClient

from src.api.logs import configure_logging
from src.api.main import app
from src.api.schemas import EXAMPLE_TRANSACTION, Transaction

# The app loads its model from MLflow at startup: use the temporary test store.
pytestmark = pytest.mark.usefixtures("project")


class FixedModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, transactions: list[Transaction]) -> list[float]:
        return [self.probability] * len(transactions)


class BrokenModel:
    def predict_proba(self, transactions: list[Transaction]) -> list[float]:
        raise RuntimeError("model failed")


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client


def read_logs(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def find(logs: list[dict[str, Any]], event: str) -> dict[str, Any]:
    [log] = [log for log in logs if log["event"] == event]
    return log


def test_logs_are_json_with_level_and_timestamp(capsys: pytest.CaptureFixture[str]):
    structlog.get_logger().info("hello", answer=42)
    [log] = read_logs(capsys)
    assert log["event"] == "hello"
    assert log["answer"] == 42
    assert log["level"] == "info"
    assert "timestamp" in log


def test_log_level_filters_lower_levels(capsys: pytest.CaptureFixture[str]):
    configure_logging("warning")
    try:
        structlog.get_logger().info("hidden")
        structlog.get_logger().warning("shown")
        assert [log["event"] for log in read_logs(capsys)] == ["shown"]
    finally:
        configure_logging("INFO")


def test_startup_logs_model_state(capsys: pytest.CaptureFixture[str], client: TestClient):
    assert find(read_logs(capsys), "startup")["model_loaded"] is False


def test_request_is_logged(capsys: pytest.CaptureFixture[str], client: TestClient):
    client.get("/health")
    log = find(read_logs(capsys), "request_completed")
    assert log["method"] == "GET"
    assert log["path"] == "/health"
    assert log["status_code"] == 200
    assert log["duration_ms"] >= 0


def test_request_id_is_generated_and_returned(capsys: pytest.CaptureFixture[str], client: TestClient):
    response = client.get("/health")
    request_id = response.headers["X-Request-ID"]
    assert find(read_logs(capsys), "request_completed")["request_id"] == request_id


def test_request_id_from_client_tags_all_logs(capsys: pytest.CaptureFixture[str], client: TestClient):
    app.state.model = FixedModel(0.9)
    response = client.post(
        "/predict", json=EXAMPLE_TRANSACTION, headers={"X-Request-ID": "abc-123"}
    )
    assert response.headers["X-Request-ID"] == "abc-123"
    logs = read_logs(capsys)
    assert find(logs, "prediction")["request_id"] == "abc-123"
    assert find(logs, "request_completed")["request_id"] == "abc-123"


def test_prediction_is_logged_without_transaction_data(capsys: pytest.CaptureFixture[str], client: TestClient):
    app.state.model = FixedModel(0.9)
    client.post("/predict", json=EXAMPLE_TRANSACTION)
    log = find(read_logs(capsys), "prediction")
    assert log["fraud_probability"] == 0.9
    assert log["is_fraud"] is True
    assert log["threshold"] == 0.5
    assert "transaction_amount" not in log


def test_batch_prediction_is_logged(capsys: pytest.CaptureFixture[str], client: TestClient):
    app.state.model = FixedModel(0.9)
    client.post("/predict/batch", json={"transactions": [EXAMPLE_TRANSACTION] * 3})
    log = find(read_logs(capsys), "batch_prediction")
    assert log["count"] == 3
    assert log["fraud_count"] == 3


def test_missing_model_is_logged(capsys: pytest.CaptureFixture[str], client: TestClient):
    client.post("/predict", json=EXAMPLE_TRANSACTION)
    log = find(read_logs(capsys), "model_not_loaded")
    assert log["level"] == "warning"


def test_unhandled_error_is_logged_with_traceback(capsys: pytest.CaptureFixture[str], client: TestClient):
    app.state.model = BrokenModel()
    with pytest.raises(RuntimeError):
        client.post("/predict", json=EXAMPLE_TRANSACTION)
    log = find(read_logs(capsys), "request_failed")
    assert log["level"] == "error"
    assert log["exception"][0]["exc_value"] == "model failed"
