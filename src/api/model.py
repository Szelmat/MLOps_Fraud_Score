from typing import Any, Protocol

import pandas as pd

from src.api.schemas import Transaction


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
