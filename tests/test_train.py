from pathlib import Path

import mlflow.sklearn as mlflow_sklearn
import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression

from src.api.schemas import EXAMPLE_TRANSACTION, Transaction
from src.ml import config, prepare, train
from tests.conftest import PARAMS, make_transactions


def test_main_logs_run_and_registers_model(project: Path):
    prepare.main()
    train.main()

    run_id = config.MLFLOW_RUN_ID_PATH.read_text()
    client = MlflowClient(config.MLFLOW_TRACKING_URI)
    run = client.get_run(run_id)
    assert run.data.params["train.model"] == "random_forest"
    assert 0 <= run.data.metrics["f1"] <= 1
    assert run.data.tags["dvc_data_hash"] == "unknown"
    assert run.inputs.dataset_inputs[0].dataset.name == "train.csv"
    [version] = client.search_model_versions(f"name='{config.MLFLOW_REGISTERED_MODEL}'")
    assert version.run_id == run_id


def test_registered_pipeline_scores_an_api_request(project: Path):
    """The API sends Transaction.to_feature_row(), including Timestamp; the Pipeline must accept it."""
    prepare.main()
    train.main()
    model = mlflow_sklearn.load_model(f"models:/{config.MLFLOW_REGISTERED_MODEL}/1")
    request = pd.DataFrame([Transaction.model_validate(EXAMPLE_TRANSACTION).to_feature_row()])
    [probability] = model.predict_proba(request)[:, 1]
    assert 0 <= probability <= 1


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
