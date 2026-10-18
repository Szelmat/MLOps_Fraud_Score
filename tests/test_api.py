import pytest
from fastapi.testclient import TestClient

from src.api.config import Settings
from src.api.schemas import EXAMPLE_TRANSACTION, Transaction
from src.api.main import app


class FakeModel:
    """Returns a fixed probability per transaction, keyed by amount."""

    def __init__(self, probabilities: dict[float, float]):
        self.probabilities = probabilities

    def predict_proba(self, transactions: list[Transaction]) -> list[float]:
        return [self.probabilities[t.transaction_amount] for t in transactions]


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def model(client):
    model = FakeModel({39.85: 0.9, 10.0: 0.1})
    app.state.model = model
    return model


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health_without_model(client):
    response = client.get("/health")
    assert response.json() == {"status": "ok", "model_loaded": False}


def test_health_with_model(client, model):
    response = client.get("/health")
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_without_model_returns_503(client):
    response = client.post("/predict", json=EXAMPLE_TRANSACTION)
    assert response.status_code == 503
    assert response.json() == {"detail": "Model is not loaded"}


def test_predict(client, model):
    response = client.post("/predict", json=EXAMPLE_TRANSACTION)
    assert response.status_code == 200
    assert response.json() == {
        "fraud_probability": 0.9,
        "is_fraud": True,
        "threshold": 0.5,
    }


def test_predict_uses_threshold_from_env(monkeypatch):
    monkeypatch.setenv("FRAUD_THRESHOLD", "0.95")
    with TestClient(app) as client:
        app.state.model = FakeModel({39.85: 0.9})
        response = client.post("/predict", json=EXAMPLE_TRANSACTION)
    assert response.json()["is_fraud"] is False
    assert response.json()["threshold"] == 0.95


@pytest.mark.parametrize(
    "change",
    [
        {"transaction_type": "Cheque"},
        {"risk_score": 1.5},
        {"transaction_amount": -1},
        {"unknown_field": 1},
    ],
)
def test_predict_rejects_invalid_input(client, model, change):
    response = client.post("/predict", json=EXAMPLE_TRANSACTION | change)
    assert response.status_code == 422


def test_predict_rejects_missing_field(client, model):
    transaction = dict(EXAMPLE_TRANSACTION)
    del transaction["card_type"]
    response = client.post("/predict", json=transaction)
    assert response.status_code == 422


def test_predict_batch_keeps_order(client, model):
    transactions = [
        EXAMPLE_TRANSACTION | {"transaction_amount": 10.0},
        EXAMPLE_TRANSACTION,
    ]
    response = client.post("/predict/batch", json={"transactions": transactions})
    assert response.status_code == 200
    assert [p["fraud_probability"] for p in response.json()["predictions"]] == [0.1, 0.9]
    assert [p["is_fraud"] for p in response.json()["predictions"]] == [False, True]


def test_predict_batch_rejects_empty(client, model):
    response = client.post("/predict/batch", json={"transactions": []})
    assert response.status_code == 422


def test_cors_allows_frontend_dev_server(client):
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_origins_from_env(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com, https://b.example.com,")
    assert Settings().cors_origins == ["https://app.example.com", "https://b.example.com"]


def test_to_feature_row_uses_dataset_columns_and_formats():
    row = Transaction(**EXAMPLE_TRANSACTION).to_feature_row()
    assert row["Timestamp"] == "2023-01-22 23:47:00"
    assert row["IP_Address_Flag"] == 1
    assert row["Is_Weekend"] == 1
    assert row["Previous_Fraudulent_Activity"] == 0
    assert row["Transaction_Type"] == "POS"
    assert len(row) == 18
