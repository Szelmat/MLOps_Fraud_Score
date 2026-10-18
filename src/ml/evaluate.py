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
from mlflow import MlflowClient
from numpy.typing import ArrayLike

from src.ml import config, scoring

log = config.get_logger(__name__)


def should_promote(f1: float, champion_f1: float | None, min_f1: float) -> bool:
    """The promotion rule: above the floor, and not worse than the champion (if there is one)."""
    return f1 >= min_f1 and (champion_f1 is None or f1 >= champion_f1)


def champion_f1(client: MlflowClient, X: pd.DataFrame, y: ArrayLike, threshold: float) -> float | None:
    """The champion's F1 re-scored on the same validation data, or None before the first promotion."""
    name = config.MLFLOW_REGISTERED_MODEL
    if config.CHAMPION_ALIAS not in client.get_registered_model(name).aliases:
        return None
    champion = mlflow_sklearn.load_model(f"models:/{name}@{config.CHAMPION_ALIAS}")
    return scoring.compute_metrics(champion, X, y, threshold)["f1"]


def main() -> dict[str, Any]:
    params: dict[str, Any] = config.load_params()
    target = params["prepare"]["target"]
    evaluate_params = params["evaluate"]

    run_id = config.MLFLOW_RUN_ID_PATH.read_text().strip()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient(config.MLFLOW_TRACKING_URI)

    valid_df = pd.read_csv(config.VALID_CSV)
    X, y = valid_df.drop(columns=[target]), valid_df[target]
    pipe = mlflow_sklearn.load_model(f"runs:/{run_id}/model")
    metrics = scoring.compute_metrics(pipe, X, y, evaluate_params["threshold"])
    config.write_json(config.METRICS_PATH, metrics)

    best = champion_f1(client, X, y, evaluate_params["threshold"])
    promoted = should_promote(metrics["f1"], best, evaluate_params["min_f1"])
    [model_version] = client.search_model_versions(f"run_id='{run_id}'")
    version = str(model_version.version)
    if promoted:
        client.set_registered_model_alias(config.MLFLOW_REGISTERED_MODEL, config.CHAMPION_ALIAS, version)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({"threshold": evaluate_params["threshold"], "min_f1": evaluate_params["min_f1"]})
        mlflow.log_metrics(metrics)
        if best is not None:
            mlflow.log_metric("champion_f1", best)
        mlflow.set_tag("promoted", str(promoted))
        mlflow.log_artifact(str(config.METRICS_PATH))

    log.info("version=%s f1=%.4f champion_f1=%s promoted=%s", version, metrics["f1"], best, promoted)
    return {"run_id": run_id, "model_version": version, "metrics": metrics, "promoted": promoted}


if __name__ == "__main__":  # pragma: no cover - script entry point; main() is tested directly
    main()
