from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.config import Settings
from api.model import FraudModel
from api.schemas import BatchPrediction, BatchRequest, Health, Prediction, Transaction


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = Settings()
    # Set to an MLflow-backed FraudModel once the model is registered there.
    app.state.model = None
    yield


app = FastAPI(title="MLOps Fraud Score", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Settings().cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_model(request: Request) -> FraudModel:
    model = request.app.state.model
    if model is None:
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
    return to_prediction(probability, threshold)


@app.post("/predict/batch")
def predict_batch(
    batch: BatchRequest, model: Model, threshold: Threshold
) -> BatchPrediction:
    probabilities = model.predict_proba(batch.transactions)
    return BatchPrediction(predictions=[to_prediction(p, threshold) for p in probabilities])
