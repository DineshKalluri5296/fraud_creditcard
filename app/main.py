import time
import json
from app.drift import calculate_drift
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from app.predict import predict_fraud

app = FastAPI(
    title="Fraud Detection API",
    version="1.0"
)

# Prometheus Metrics
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total Prediction Requests"
)

FRAUD_COUNT = Counter(
    "fraud_predictions_total",
    "Total Fraud Predictions"
)

NON_FRAUD_COUNT = Counter(
    "nonfraud_predictions_total",
    "Total Non-Fraud Predictions"
)

MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Model Accuracy"
)

DATA_DRIFT_SCORE = Gauge(
    "data_drift_score",
    "Current Data Drift Score"
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction Latency"
)

# Set your trained model accuracy (optional)
try:
    with open("model/model_metrics.json") as f:
        metrics = json.load(f)
    MODEL_ACCURACY.set(metrics["accuracy"])
except:
    MODEL_ACCURACY.set(0)


class FraudRequest(BaseModel):
    distance_from_home: float
    distance_from_last_transaction: float
    ratio_to_median_purchase_price: float
    repeat_retailer: float
    used_chip: float
    used_pin_number: float
    online_order: float


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API Running"
    }


@app.post("/predict")
def predict(request: FraudRequest):
    REQUEST_COUNT.inc()

    start = time.time()

    values = [
        request.distance_from_home,
        request.distance_from_last_transaction,
        request.ratio_to_median_purchase_price,
        request.repeat_retailer,
        request.used_chip,
        request.used_pin_number,
        request.online_order,
    ]

    # Model prediction
    prediction, probability = predict_fraud(values)
    drift = calculate_drift(values)
    DATA_DRIFT_SCORE.set(drift)

    if prediction == 1:
        FRAUD_COUNT.inc()
    else:
        NON_FRAUD_COUNT.inc()

    PREDICTION_LATENCY.observe(time.time() - start)

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": round(float(probability), 4),
    }


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )
