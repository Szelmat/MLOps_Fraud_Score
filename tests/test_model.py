import numpy as np
import pandas as pd

from src.api.model import MlflowFraudModel
from src.api.schemas import EXAMPLE_TRANSACTION, Transaction


class RecordingPipeline:
    """Stands in for the sklearn Pipeline: remembers its input, returns fixed probabilities."""

    def __init__(self) -> None:
        self.seen = pd.DataFrame()

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        self.seen = rows
        return np.array([[0.25, 0.75]] * len(rows))


def test_model_sends_raw_feature_rows_to_the_pipeline():
    pipeline = RecordingPipeline()
    transaction = Transaction.model_validate(EXAMPLE_TRANSACTION)
    assert MlflowFraudModel(pipeline).predict_proba([transaction, transaction]) == [0.75, 0.75]
    assert list(pipeline.seen.columns) == list(transaction.to_feature_row())
