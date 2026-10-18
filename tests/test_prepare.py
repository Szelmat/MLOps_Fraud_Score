from pathlib import Path

import pandas as pd
import pytest

from src.ml import config
from src.ml.prepare import clean, main
from tests.conftest import make_transactions

DROP = ["Transaction_ID", "User_ID", "Timestamp"]


def test_clean_drops_configured_columns():
    cleaned = clean(make_transactions(50), "Fraud_Label", DROP)
    assert not set(DROP) & set(cleaned.columns)
    assert len(cleaned) == 50


def test_clean_drops_duplicates_hidden_by_ids():
    raw = make_transactions(50)
    duplicate = raw.iloc[[0]].assign(Transaction_ID="TXN_copy")
    assert len(clean(pd.concat([raw, duplicate]), "Fraud_Label", DROP)) == 50


def test_clean_rejects_non_binary_target():
    raw = make_transactions(50)
    raw.loc[0, "Fraud_Label"] = 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        clean(raw, "Fraud_Label", DROP)


def test_main_writes_stratified_splits(project: Path):
    main()
    train, valid = pd.read_csv(config.TRAIN_CSV), pd.read_csv(config.VALID_CSV)
    assert (len(train), len(valid)) == (300, 100)
    assert abs(train["Fraud_Label"].mean() - valid["Fraud_Label"].mean()) < 0.02
