"""Metric computation -- ONE implementation, used by both train.py and evaluate.py.

If the training script and the evaluation script compute metrics differently,
you will spend a day chasing a difference that does not exist. One function.
"""

from typing import Any

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# zero_division=0: score 0 instead of warning when nothing is predicted positive. Passed as
# kwargs because sklearn leaves it unannotated and pyright infers `str` from its default "warn".
ZERO_DIVISION: dict[str, Any] = {"zero_division": 0}


def compute_metrics(pipe: Any, X: pd.DataFrame, y: ArrayLike, threshold: float = 0.5) -> dict[str, float]:
    """Metrics of a fitted Pipeline (or anything with predict_proba) at the decision threshold."""
    proba = pipe.predict_proba(X)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "precision": round(float(precision_score(y, y_pred, **ZERO_DIVISION)), 4),
        "recall": round(float(recall_score(y, y_pred, **ZERO_DIVISION)), 4),
        "f1": round(float(f1_score(y, y_pred, **ZERO_DIVISION)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
    }
