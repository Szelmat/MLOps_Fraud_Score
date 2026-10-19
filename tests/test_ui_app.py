from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.ui import api_client
from src.ui.api_client import ApiError, Health, TrainResult

APP = str(Path(__file__).parents[1] / "src" / "ui" / "app.py")
RESULT: TrainResult = {
    "run_id": "abc123",
    "model_version": "2",
    "metrics": {"f1": 0.7865, "roc_auc": 0.8436},
    "promoted": True,
}


class FakeClient:
    """Stands in for ApiClient; counts the requests the app makes."""

    def __init__(self, health: Health | ApiError, train: TrainResult | ApiError = RESULT) -> None:
        self.health_answers = [health]
        self.train_answer = train
        self.health_calls = 0
        self.train_calls = 0

    def health(self) -> Health:
        self.health_calls += 1
        answer = self.health_answers[min(self.health_calls, len(self.health_answers)) - 1]
        if isinstance(answer, ApiError):
            raise answer
        return answer

    def train(self) -> TrainResult:
        self.train_calls += 1
        if isinstance(self.train_answer, ApiError):
            raise self.train_answer
        return self.train_answer


@pytest.fixture(autouse=True)
def clear_client_cache() -> Iterator[None]:
    """The app caches its client across sessions; each test gets its own fake."""
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def run_app(monkeypatch: pytest.MonkeyPatch, client: FakeClient) -> AppTest:
    monkeypatch.setattr(api_client, "ApiClient", lambda: client)
    return AppTest.from_file(APP, default_timeout=10).run()


def texts(app: AppTest) -> str:
    return " ".join(m.value for m in [*app.markdown, *app.caption])


def test_header_shows_model_loaded(monkeypatch: pytest.MonkeyPatch):
    app = run_app(monkeypatch, FakeClient({"status": "ok", "model_loaded": True}))
    assert app.title[0].value == "MLOps Fraud Score"
    assert "API online · model loaded" in texts(app)
    assert not app.button[0].disabled


def test_header_shows_missing_model(monkeypatch: pytest.MonkeyPatch):
    app = run_app(monkeypatch, FakeClient({"status": "ok", "model_loaded": False}))
    assert "API online · no model" in texts(app)
    assert "Run a training" in texts(app)


def test_unreachable_api_disables_training(monkeypatch: pytest.MonkeyPatch):
    app = run_app(monkeypatch, FakeClient(ApiError("Cannot reach the API at http://api")))
    assert "API unreachable" in texts(app)
    assert "Cannot reach the API at http://api" in texts(app)
    assert app.button[0].disabled


def test_health_is_requested_once_per_session(monkeypatch: pytest.MonkeyPatch):
    client = FakeClient({"status": "ok", "model_loaded": True})
    app = run_app(monkeypatch, client)
    app.run()
    app.run()
    assert client.health_calls == 1


def test_training_shows_metrics_and_refreshes_health(monkeypatch: pytest.MonkeyPatch):
    client = FakeClient({"status": "ok", "model_loaded": False})
    client.health_answers.append({"status": "ok", "model_loaded": True})
    app = run_app(monkeypatch, client)
    app.button[0].click().run()
    assert client.train_calls == 1
    assert "passed the gate" in app.success[0].value
    assert [m.label for m in app.metric] == ["F1", "ROC AUC"]
    assert [m.value for m in app.metric] == ["0.7865", "0.8436"]
    assert "abc123" in texts(app)
    assert client.health_calls == 2
    assert "API online · model loaded" in texts(app)


def test_rejected_model_shows_warning(monkeypatch: pytest.MonkeyPatch):
    client = FakeClient({"status": "ok", "model_loaded": True}, train={**RESULT, "promoted": False})
    app = run_app(monkeypatch, client)
    app.button[0].click().run()
    assert "did not pass the gate" in app.warning[0].value


def test_training_error_is_shown(monkeypatch: pytest.MonkeyPatch):
    client = FakeClient({"status": "ok", "model_loaded": True}, train=ApiError("409: Training is already running"))
    app = run_app(monkeypatch, client)
    app.button[0].click().run()
    assert app.error[0].value == "Training failed: 409: Training is already running"
    assert not app.success
