from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Fraud Detection API Running"


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200


def test_predict(monkeypatch):
    def mock_predict(_):
        return 0, 0.001

    # Patch the imported function in app.main
    monkeypatch.setattr("app.main.predict_fraud", mock_predict)

    payload = {
        "distance_from_home": 10.5,
        "distance_from_last_transaction": 2.1,
        "ratio_to_median_purchase_price": 1.3,
        "repeat_retailer": 1,
        "used_chip": 1,
        "used_pin_number": 0,
        "online_order": 1
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["fraud_prediction"] == 0
