from src.ml.prepare import clean
from tests.conftest import make_transactions

DROP = ["Transaction_ID", "User_ID", "Timestamp"]


def test_clean_drops_configured_columns():
    cleaned = clean(make_transactions(50), "Fraud_Label", DROP)
    assert not set(DROP) & set(cleaned.columns)
    assert len(cleaned) == 50
