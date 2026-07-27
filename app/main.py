from fastapi import FastAPI
from pydantic import BaseModel

from prometheus_client import Counter
from prometheus_client import generate_latest
from fastapi.responses import Response

from app.predict import predict_fraud

app = FastAPI(
    title="Fraud Detection API",
    version="1.0"
)

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total Prediction Requests"
)


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

    values = [

        request.distance_from_home,
        request.distance_from_last_transaction,
        request.ratio_to_median_purchase_price,
        request.repeat_retailer,
        request.used_chip,
        request.used_pin_number,
        request.online_order

    ]

    prediction, probability = predict_fraud(values)

    return {

        "fraud_prediction": prediction,
        "fraud_probability": round(probability, 4)

    }


@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type="text/plain"
    )