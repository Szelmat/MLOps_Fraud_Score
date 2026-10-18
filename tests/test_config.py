import json
import logging
from pathlib import Path

from src.ml import config


def test_repository_params_file():
    params = config.load_params()
    assert params["prepare"]["target"] == "Fraud_Label"
    assert params["train"]["model"] in {"random_forest", "logistic_regression"}


def test_flatten_nested_params():
    assert config.flatten({"seed": 1, "train": {"model": "rf", "deep": {"x": 2}}}) == {
        "seed": 1,
        "train.model": "rf",
        "train.deep.x": 2,
    }


def test_write_json_creates_parent_dirs(tmp_path: Path):
    path = tmp_path / "nested" / "out.json"
    config.write_json(path, {"b": 1, "a": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 2, "b": 1}


def test_get_logger():
    assert isinstance(config.get_logger("x"), logging.Logger)
