import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.ml.scoring import compute_metrics


class FixedProba:
    """Stands in for a Pipeline: returns preset fraud probabilities."""

    def __init__(self, proba: list[float]):
        self.proba = np.array(proba)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([1 - self.proba, self.proba])


X = pd.DataFrame({"a": range(5)})
Y = pd.Series([0, 0, 0, 1, 1])


def test_compute_metrics_at_threshold():
    metrics = compute_metrics(FixedProba([0.1, 0.2, 0.6, 0.7, 0.9]), X, Y, threshold=0.5)
    assert metrics["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["recall"] == 1.0
    assert metrics["accuracy"] == 0.8
    assert metrics["roc_auc"] == 1.0


def test_compute_metrics_without_positive_predictions():
    pipe = Pipeline([("model", DummyClassifier(strategy="most_frequent"))]).fit(X, Y)
    metrics = compute_metrics(pipe, X, Y)
    assert metrics["precision"] == 0.0
    assert metrics["f1"] == 0.0
