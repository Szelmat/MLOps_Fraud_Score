import json
from pathlib import Path

import pandas as pd
import pytest
from mlflow import MlflowClient

from src.ml import config, evaluate, prepare, train


@pytest.mark.parametrize(
    ("f1", "champion", "promote"),
    [(0.5, None, False), (0.7, None, True), (0.7, 0.8, False), (0.8, 0.8, True), (0.9, 0.8, True)],
)
def test_should_promote(f1: float, champion: float | None, promote: bool):
    assert evaluate.should_promote(f1, champion, min_f1=0.6) is promote


def run_pipeline() -> dict[str, object]:
    prepare.main()
    train.main()
    return evaluate.main()


def test_first_model_becomes_champion(project: Path):
    result = run_pipeline()
    assert result["promoted"] is True
    assert result["model_version"] == "1"
    client = MlflowClient(config.MLFLOW_TRACKING_URI)
    assert client.get_registered_model(config.MLFLOW_REGISTERED_MODEL).aliases == {"champion": 1}
    run = client.get_run(str(result["run_id"]))
    assert run.data.tags["promoted"] == "True"
    assert "metrics.json" in {artifact.path for artifact in client.list_artifacts(run.info.run_id)}
    assert json.loads(config.METRICS_PATH.read_text(encoding="utf-8")) == result["metrics"]


def test_worse_model_does_not_replace_the_champion(project: Path):
    run_pipeline()
    # Shuffled training labels: the new model learns noise and must not become champion.
    prepare.main()
    train_df = pd.read_csv(config.TRAIN_CSV)
    train_df["Fraud_Label"] = train_df["Fraud_Label"].sample(frac=1, random_state=0).to_numpy()
    train_df.to_csv(config.TRAIN_CSV, index=False)
    train.main()
    result = evaluate.main()

    assert result["promoted"] is False
    client = MlflowClient(config.MLFLOW_TRACKING_URI)
    assert client.get_registered_model(config.MLFLOW_REGISTERED_MODEL).aliases == {"champion": 1}
    assert "champion_f1" in client.get_run(str(result["run_id"])).data.metrics
