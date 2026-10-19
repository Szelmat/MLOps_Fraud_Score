import json
import logging
from pathlib import Path

import pytest

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


def test_git_sha_in_the_repository():
    assert len(config.git_sha().removesuffix("-dirty")) == 40


def test_git_sha_outside_a_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    assert config.git_sha() == "unknown"


def test_dvc_data_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    lock = tmp_path / "dvc.lock"
    monkeypatch.setattr(config, "DVC_LOCK_PATH", lock)
    assert config.dvc_data_hash() == "unknown"
    lock.write_text(
        "stages:\n  prepare:\n    deps:\n"
        "    - path: src/x.py\n      md5: aaa\n"
        "    - path: data/raw/fraud_5000.csv\n      md5: bbb\n",
        encoding="utf-8",
    )
    assert config.dvc_data_hash() == "bbb"
    assert config.dvc_data_hash("other.csv") == "unknown"


def test_lineage_tags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "DVC_LOCK_PATH", tmp_path / "dvc.lock")
    assert config.lineage_tags() == {"git_sha": "unknown", "dvc_data_hash": "unknown"}
