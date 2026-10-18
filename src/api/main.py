import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import Settings
from src.api.logs import configure_logging
from src.api.model import FraudModel
from src.api.schemas import BatchPrediction, BatchRequest, Health, Prediction, Transaction

settings = Settings()
configure_logging(settings.log_level)
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = Settings()
    # Set to an MLflow-backed FraudModel once the model is registered there.
    app.state.model = None
    logger.info("startup", model_loaded=app.state.model is not None)
    yield


app = FastAPI(title="MLOps Fraud Score", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Logs every request and tags all logs it produces with a request ID.

    The ID comes from the client's `X-Request-ID` header if present, and is
    returned in the response's `X-Request-ID` header.
    """
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id, method=request.method, path=request.url.path
    )
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed", duration_ms=_elapsed_ms(start))
        raise
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_completed",
        status_code=response.status_code,
        duration_ms=_elapsed_ms(start),
    )
    return response


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def get_model(request: Request) -> FraudModel:
    model = request.app.state.model
    if model is None:
        logger.warning("model_not_loaded")
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return model


def get_threshold(request: Request) -> float:
    return request.app.state.settings.fraud_threshold


Model = Annotated[FraudModel, Depends(get_model)]
Threshold = Annotated[float, Depends(get_threshold)]


def to_prediction(probability: float, threshold: float) -> Prediction:
    return Prediction(
        fraud_probability=probability,
        is_fraud=probability >= threshold,
        threshold=threshold,
    )


@app.get("/")
def read_root():
    return {"project": "MLOps Fraud Score"}


@app.get("/health")
def health(request: Request) -> Health:
    return Health(status="ok", model_loaded=request.app.state.model is not None)


@app.post("/predict")
def predict(transaction: Transaction, model: Model, threshold: Threshold) -> Prediction:
    [probability] = model.predict_proba([transaction])
    prediction = to_prediction(probability, threshold)
    logger.info(
        "prediction",
        fraud_probability=prediction.fraud_probability,
        is_fraud=prediction.is_fraud,
        threshold=threshold,
    )
    return prediction


@app.post("/predict/batch")
def predict_batch(
    batch: BatchRequest, model: Model, threshold: Threshold
) -> BatchPrediction:
    probabilities = model.predict_proba(batch.transactions)
    predictions = [to_prediction(p, threshold) for p in probabilities]
    logger.info(
        "batch_prediction",
        count=len(predictions),
        fraud_count=sum(p.is_fraud for p in predictions),
        threshold=threshold,
    )
    return BatchPrediction(predictions=predictions)
