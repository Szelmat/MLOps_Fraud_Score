import httpx
import pytest

from src.ui.api_client import DEFAULT_API_URL, ApiClient, ApiError

TRAIN_RESULT = {"run_id": "abc", "model_version": "2", "metrics": {"f1": 0.79}, "promoted": True}


def client_for(transport: httpx.MockTransport) -> ApiClient:
    return ApiClient("http://api", transport=transport)


def test_health_returns_the_api_answer():
    def handler(request: httpx.Request) -> httpx.Response:
        assert (request.method, request.url.path) == ("GET", "/health")
        return httpx.Response(200, json={"status": "ok", "model_loaded": True})

    assert client_for(httpx.MockTransport(handler)).health() == {"status": "ok", "model_loaded": True}


def test_train_posts_and_returns_the_result():
    def handler(request: httpx.Request) -> httpx.Response:
        assert (request.method, request.url.path) == ("POST", "/train")
        return httpx.Response(200, json=TRAIN_RESULT)

    assert client_for(httpx.MockTransport(handler)).train() == TRAIN_RESULT


def test_error_status_raises_with_the_detail():
    transport = httpx.MockTransport(lambda _: httpx.Response(409, json={"detail": "Training is already running"}))
    with pytest.raises(ApiError, match="409: Training is already running"):
        client_for(transport).train()


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(500, text="boom"), "500: boom"),
        (httpx.Response(502), "502: Bad Gateway"),
        (httpx.Response(500, json={"message": "boom"}), '500: {"message":"boom"}'),
        (httpx.Response(500, json=["boom"]), '500: \\["boom"\\]'),
    ],
)
def test_error_without_detail_falls_back_to_the_body(response: httpx.Response, message: str):
    with pytest.raises(ApiError, match=message):
        client_for(httpx.MockTransport(lambda _: response)).health()


def test_unreachable_api_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ApiError, match="Cannot reach the API at http://api"):
        client_for(httpx.MockTransport(handler)).health()


def test_base_url_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("API_URL", "http://api:8000")
    assert ApiClient().base_url == "http://api:8000"
    monkeypatch.delenv("API_URL")
    assert ApiClient().base_url == DEFAULT_API_URL
