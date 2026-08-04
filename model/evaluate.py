import os
import json
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score

MODEL_PATH = "model/model.pkl"

THRESHOLD = 0.95

if not os.path.exists(MODEL_PATH):
    print("Model not found")
    exit(1)

model = joblib.load(MODEL_PATH)

df = pd.read_csv("data/card_transdata.csv")

X = df.drop("fraud", axis=1)

y = df["fraud"]

pred = model.predict(X)

accuracy = accuracy_score(y, pred)

print(f"Accuracy : {accuracy:.4f}")

with open("model/model_metrics.json", "w") as f:
    json.dump({"accuracy": float(accuracy)}, f, indent=4)

if accuracy < THRESHOLD:

    print("Accuracy dropped below threshold")

    exit(1)

print("Accuracy acceptable")

exit(0)
