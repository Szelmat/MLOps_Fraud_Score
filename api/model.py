from typing import Protocol

from api.schemas import Transaction


class FraudModel(Protocol):
    """What the API needs from a fraud model.

    The trained model will be served from MLflow; an adapter around it
    implements this interface. Until one is attached to `app.state.model`,
    the prediction endpoints return 503.
    """

    def predict_proba(self, transactions: list[Transaction]) -> list[float]:
        """Returns the probability of fraud for each transaction, in order."""
        ...
