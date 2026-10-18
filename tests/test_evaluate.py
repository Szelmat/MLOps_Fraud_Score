import json
from pathlib import Path

import pytest
from mlflow import MlflowClient

from src.ml import config, evaluate, prepare, train


@pytest.mark.parametrize(
    ("f1", "champion", "promote"),
    [(0.5, None, False), (0.7, None, True), (0.7, 0.8, False), (0.8, 0.8, True), (0.9, 0.8, True)],
)
def test_should_promote(f1: float, champion: float | None, promote: bool):
    assert evaluate.should_promote(f1, champion, min_f1=0.6) is promote


def test_evaluation_is_logged_to_the_training_run(project: Path):
    prepare.main()
    train.main()
    result = evaluate.main()
    client = MlflowClient(config.MLFLOW_TRACKING_URI)
    run = client.get_run(str(result["run_id"]))
    assert "metrics.json" in {artifact.path for artifact in client.list_artifacts(run.info.run_id)}
    assert json.loads(config.METRICS_PATH.read_text(encoding="utf-8")) == result["metrics"]
