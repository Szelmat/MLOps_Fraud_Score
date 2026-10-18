"""DVC stage 2 -- train: fit a scikit-learn Pipeline, track the run in MLflow, register the model.

The preprocessing lives INSIDE the Pipeline on purpose: that is what removes the
training/serving skew class of bugs. The API sends raw columns, the Pipeline does the rest.

Run:  python -m src.ml.train      (or: dvc repro train)
"""

from datetime import datetime
from importlib.metadata import version
from typing import Any

import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd
from mlflow.data.pandas_dataset import from_pandas
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml import config, scoring

log = config.get_logger(__name__)

# MLflow saves sklearn models with skops instead of pickle; skops only loads the types listed here,
# so a tampered model file cannot run arbitrary code when the API loads it.
SKOPS_TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]
# Pinned so MLflow doesn't infer them by loading the model in a slow subprocess.
MODEL_REQUIREMENTS = [f"{package}=={version(package)}" for package in ("scikit-learn", "pandas", "numpy", "skops")]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """One-hot for text columns, scaling for numeric ones."""
    categorical: list[str] = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    numeric: list[str] = [c for c in X.columns if c not in categorical]
    log.info("categorical=%d numeric=%d", len(categorical), len(numeric))
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        # handle_unknown='ignore': unseen categories at serving time must not crash the API.
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ],
        # Columns the model wasn't trained on (e.g. Timestamp from the API) are dropped.
        remainder="drop",
    )


def build_estimator(
    model_name: str, seed: int, train_params: dict[str, Any]
) -> RandomForestClassifier | LogisticRegression:
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=train_params["n_estimators"],
            max_depth=train_params["max_depth"],
            min_samples_leaf=train_params["min_samples_leaf"],
            class_weight=train_params["class_weight"],
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight=train_params["class_weight"], random_state=seed)
    raise ValueError(f"Unknown model: {model_name!r}")


def build_pipeline(X: pd.DataFrame, model_name: str, seed: int, train_params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(X)),
            ("model", build_estimator(model_name, seed, train_params)),
        ]
    )


def main() -> None:
    params: dict[str, Any] = config.load_params()
    seed: int = params["seed"]
    target: str = params["prepare"]["target"]
    train_params: dict[str, Any] = params["train"]

    train_df = pd.read_csv(config.TRAIN_CSV)
    valid_df = pd.read_csv(config.VALID_CSV)
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_valid, y_valid = valid_df.drop(columns=[target]), valid_df[target]

    pipeline = build_pipeline(X_train, train_params["model"], seed, train_params)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{train_params['model']} {datetime.now():%m%d_%H%M}") as run:
        mlflow.log_input(from_pandas(train_df, name=config.TRAIN_CSV.name, targets=target), context="training")
        mlflow.log_params(config.flatten(params))
        mlflow.set_tags(config.lineage_tags())

        pipeline.fit(X_train, y_train)

        metrics = scoring.compute_metrics(pipeline, X_valid, y_valid, params["evaluate"]["threshold"])
        mlflow.log_metrics(metrics)

        mlflow_sklearn.log_model(
            pipeline,
            name="model",
            input_example=X_train.head(5),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
            pip_requirements=MODEL_REQUIREMENTS,
        )

        config.ensure_dirs(config.MLFLOW_RUN_ID_PATH.parent)
        config.MLFLOW_RUN_ID_PATH.write_text(run.info.run_id)
        log.info("MLflow run_id: %s metrics: %s", run.info.run_id, metrics)


if __name__ == "__main__":  # pragma: no cover - script entry point; main() is tested directly
    main()
