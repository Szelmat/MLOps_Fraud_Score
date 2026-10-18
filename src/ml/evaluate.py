"""DVC stage 3 -- evaluate: score the new model and promote it only if it is not worse.

This is the automatic guard against model quality dropping. The new version gets the
registry's "champion" alias -- the one the API serves -- only if its F1 on the validation
split clears `min_f1` AND is not below the current champion's F1 on the same split.

Writes metrics/metrics.json, declared as `metrics` in dvc.yaml with `cache: false`,
so it goes into Git and `dvc metrics diff` can compare commits.

Run:  python -m src.ml.evaluate      (or: dvc repro evaluate)
"""


from typing import Any

import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd

from src.ml import config, scoring

log = config.get_logger(__name__)


def should_promote(f1: float, champion_f1: float | None, min_f1: float) -> bool:
    """The promotion rule: above the floor, and not worse than the champion (if there is one)."""
    return f1 >= min_f1 and (champion_f1 is None or f1 >= champion_f1)


def main() -> dict[str, Any]:
    params: dict[str, Any] = config.load_params()
    target = params["prepare"]["target"]
    evaluate_params = params["evaluate"]

    run_id = config.MLFLOW_RUN_ID_PATH.read_text().strip()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    valid_df = pd.read_csv(config.VALID_CSV)
    X, y = valid_df.drop(columns=[target]), valid_df[target]
    pipe = mlflow_sklearn.load_model(f"runs:/{run_id}/model")
    metrics = scoring.compute_metrics(pipe, X, y, evaluate_params["threshold"])
    config.write_json(config.METRICS_PATH, metrics)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({"threshold": evaluate_params["threshold"], "min_f1": evaluate_params["min_f1"]})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(config.METRICS_PATH))

    log.info("run_id=%s f1=%.4f", run_id, metrics["f1"])
    return {"run_id": run_id, "metrics": metrics}


if __name__ == "__main__":  # pragma: no cover - script entry point; main() is tested directly
    main()
