import joblib
import numpy as np

# Load trained model only once
model = joblib.load("model/model.pkl")


def predict_fraud(features):

    data = np.array(features).reshape(1, -1)

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    return int(prediction), float(probability)