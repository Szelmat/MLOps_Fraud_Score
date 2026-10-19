import threading
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import Settings
from src.api.logs import configure_logging
from src.api.model import FraudModel, load_champion_model
from src.api.prediction_log import log_predictions, upload_prediction_log
from src.api.schemas import BatchPrediction, BatchRequest, Health, Prediction, Transaction, TrainResult
from src.ml import config, evaluate, prepare
from src.ml import train as train_stage

settings = Settings()
configure_logging(settings.log_level)
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    app.state.settings = Settings()
    app.state.model = load_champion_model()
    logger.info("startup", model_loaded=app.state.model is not None)
    yield
    upload_prediction_log(config.PREDICTION_LOG_PATH)


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
def read_root() -> dict[str, str]:
    return {"project": "MLOps Fraud Score"}


@app.get("/health")
def health(request: Request) -> Health:
    return Health(status="ok", model_loaded=request.app.state.model is not None)


@app.post("/predict")
def predict(transaction: Transaction, model: Model, threshold: Threshold) -> Prediction:
    [probability] = model.predict_proba([transaction])
    prediction = to_prediction(probability, threshold)
    log_predictions(config.PREDICTION_LOG_PATH, [transaction], [prediction])
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
    log_predictions(config.PREDICTION_LOG_PATH, batch.transactions, predictions)
    logger.info(
        "batch_prediction",
        count=len(predictions),
        fraud_count=sum(p.is_fraud for p in predictions),
        threshold=threshold,
    )
    return BatchPrediction(predictions=predictions)


# Two trainings at once would overwrite each other's files and race for the champion alias.
training_lock = threading.Lock()


@app.post("/train")
def train(request: Request) -> TrainResult:
    """Runs prepare -> train -> evaluate, like `dvc repro`, and serves the new model if it was promoted."""
    if not training_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Training is already running")
    try:
        prepare.main()
        train_stage.main()
        result = evaluate.main()
    finally:
        training_lock.release()
    if result["promoted"]:
        # Keep serving the old model if the new one can't be loaded.
        request.app.state.model = load_champion_model() or request.app.state.model
    logger.info("training_finished", run_id=result["run_id"], promoted=result["promoted"])
    return TrainResult(**result)
