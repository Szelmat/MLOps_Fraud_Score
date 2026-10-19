from pathlib import Path

import yaml
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from src.api.main import app, training_lock
from src.api.schemas import EXAMPLE_TRANSACTION
from src.ml import config
from tests.conftest import PARAMS


def test_train_promotes_the_first_model_and_serves_it(project: Path):
    with TestClient(app) as client:
        assert client.post("/predict", json=EXAMPLE_TRANSACTION).status_code == 503
        response = client.post("/train")
        prediction = client.post("/predict", json=EXAMPLE_TRANSACTION)
    assert response.status_code == 200
    assert response.json()["promoted"] is True
    assert response.json()["model_version"] == "1"
    assert prediction.status_code == 200
    assert 0 <= prediction.json()["fraud_probability"] <= 1
    run = MlflowClient(config.MLFLOW_TRACKING_URI).get_run(response.json()["run_id"])
    assert run.data.tags["promoted"] == "True"


def test_rejected_model_keeps_the_served_champion(project: Path):
    with TestClient(app) as client:
        client.post("/train")
        served = app.state.model
        params = PARAMS | {"evaluate": PARAMS["evaluate"] | {"min_f1": 0.99}}
        config.PARAMS_PATH.write_text(yaml.safe_dump(params), encoding="utf-8")
        response = client.post("/train")
    assert response.json()["promoted"] is False
    assert app.state.model is served


def test_only_one_training_at_a_time(project: Path):
    with TestClient(app) as client, training_lock:
        assert client.post("/train").status_code == 409
