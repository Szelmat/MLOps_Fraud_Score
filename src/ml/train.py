"""DVC stage 2 -- train: fit a scikit-learn Pipeline, track the run in MLflow, register the model.

The preprocessing lives INSIDE the Pipeline on purpose: that is what removes the
training/serving skew class of bugs. The API sends raw columns, the Pipeline does the rest.

Run:  python -m src.ml.train      (or: dvc repro train)
"""

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml import config

log = config.get_logger(__name__)


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
