from src.ml import config


def test_repository_params_file():
    params = config.load_params()
    assert params["prepare"]["target"] == "Fraud_Label"
    assert params["train"]["model"] in {"random_forest", "logistic_regression"}
