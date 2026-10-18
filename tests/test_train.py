import pytest
from sklearn.linear_model import LogisticRegression

from src.ml import train
from tests.conftest import PARAMS, make_transactions


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
