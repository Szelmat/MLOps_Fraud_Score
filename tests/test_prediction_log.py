import json
from pathlib import Path

from src.api.prediction_log import log_predictions
from src.api.schemas import EXAMPLE_TRANSACTION, Prediction, Transaction


def test_log_predictions_appends_one_json_line_per_transaction(tmp_path: Path):
    path = tmp_path / "logs" / "predictions.jsonl"
    transaction = Transaction.model_validate(EXAMPLE_TRANSACTION)
    prediction = Prediction(fraud_probability=0.8, is_fraud=True, threshold=0.5)
    log_predictions(path, [transaction, transaction], [prediction, prediction])
    log_predictions(path, [transaction], [prediction])

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3
    assert lines[0]["Transaction_Amount"] == 39.85
    assert lines[0]["IP_Address_Flag"] == 1
    assert lines[0]["fraud_probability"] == 0.8
    assert "logged_at" in lines[0]
