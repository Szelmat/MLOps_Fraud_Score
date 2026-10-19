"""HTTP client for the scoring API. The UI reaches the API only through this module."""

import os
from typing import Any, TypedDict

import httpx

DEFAULT_API_URL = "http://localhost:8000"
HEALTH_TIMEOUT_S = 5
# POST /train runs the whole pipeline synchronously.
TRAIN_TIMEOUT_S = 600


class Health(TypedDict):
    status: str
    model_loaded: bool


class TrainResult(TypedDict):
    run_id: str
    model_version: str
    metrics: dict[str, float]
    promoted: bool


class ApiError(Exception):
    """The API is unreachable or answered with an error status."""


class ApiClient:
    def __init__(self, base_url: str | None = None, transport: httpx.BaseTransport | None = None) -> None:
        self.base_url = base_url or os.getenv("API_URL", DEFAULT_API_URL)
        self._client = httpx.Client(base_url=self.base_url, transport=transport)

    def health(self) -> Health:
        return self._request("GET", "/health", HEALTH_TIMEOUT_S)

    def train(self) -> TrainResult:
        return self._request("POST", "/train", TRAIN_TIMEOUT_S)

    def _request(self, method: str, path: str, timeout: float) -> Any:
        try:
            response = self._client.request(method, path, timeout=timeout)
        except httpx.HTTPError as error:
            raise ApiError(f"Cannot reach the API at {self.base_url}") from error
        if response.is_error:
            raise ApiError(f"{response.status_code}: {_detail(response)}")
        return response.json()


def _detail(response: httpx.Response) -> str:
    """FastAPI puts the reason in `detail`; fall back to the raw body."""
    try:
        return str(response.json()["detail"])
    except (ValueError, KeyError, TypeError):
        return response.text or response.reason_phrase
