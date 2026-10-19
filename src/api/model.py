from typing import Any, Protocol

import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd
import structlog
from mlflow.exceptions import MlflowException

from src.api.schemas import Transaction
from src.ml import config

logger = structlog.get_logger()


class FraudModel(Protocol):
    """What the API needs from a fraud model.

    `MlflowFraudModel` implements it for the registry's champion. Until a model
    is attached to `app.state.model`, the prediction endpoints return 503.
    """

    def predict_proba(self, transactions: list[Transaction]) -> list[float]:
        """Returns the probability of fraud for each transaction, in order."""
        ...


class MlflowFraudModel:
    """The champion sklearn Pipeline behind the FraudModel interface."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def predict_proba(self, transactions: list[Transaction]) -> list[float]:
        # The Pipeline was trained on raw dataset columns and does its own preprocessing.
        rows = pd.DataFrame([transaction.to_feature_row() for transaction in transactions])
        return [float(p) for p in self.pipeline.predict_proba(rows)[:, 1]]


def load_champion_model() -> MlflowFraudModel | None:
    """The registry's champion, or None if there is none yet or MLflow is unreachable.

    The API must start either way; /predict answers 503 until a model is loaded.
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    try:
        pipeline = mlflow_sklearn.load_model(f"models:/{config.MLFLOW_REGISTERED_MODEL}@{config.CHAMPION_ALIAS}")
    except MlflowException:
        logger.warning("champion_not_loaded", tracking_uri=config.MLFLOW_TRACKING_URI)
        return None
    return MlflowFraudModel(pipeline)
