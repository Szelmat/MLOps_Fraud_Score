from pathlib import Path

import pytest
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression

from src.ml import config, prepare, train
from tests.conftest import PARAMS, make_transactions


def test_main_logs_run(project: Path):
    prepare.main()
    train.main()

    run_id = config.MLFLOW_RUN_ID_PATH.read_text()
    client = MlflowClient(config.MLFLOW_TRACKING_URI)
    run = client.get_run(run_id)
    assert run.data.params["train.model"] == "random_forest"
    assert 0 <= run.data.metrics["f1"] <= 1
    assert run.data.tags["dvc_data_hash"] == "unknown"
    assert run.inputs.dataset_inputs[0].dataset.name == "train.csv"


def test_pipeline_learns_from_raw_columns():
    raw = make_transactions(200).drop(columns=["Transaction_ID", "User_ID", "Timestamp"])
    X, y = raw.drop(columns=["Fraud_Label"]), raw["Fraud_Label"]
    pipeline = train.build_pipeline(X, "random_forest", 1, PARAMS["train"]).fit(X, y)
    assert pipeline.predict_proba(X).shape == (200, 2)


def test_build_estimator_logistic_regression():
    assert isinstance(train.build_estimator("logistic_regression", 1, PARAMS["train"]), LogisticRegression)


def test_build_estimator_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        train.build_estimator("svm", 1, PARAMS["train"])
