import joblib
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("data/card_transdata.csv")

# Features
X = df.drop("fraud", axis=1)

# Target
y = df["fraud"]

training_stats = {}

for col in X.columns:
    training_stats[col] = {
        "mean": float(X[col].mean()),
        "std": float(X[col].std())
    }

with open("model/training_stats.json", "w") as f:
    json.dump(training_stats, f, indent=4)

print("Training statistics saved.")



# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

# Train
model.fit(X_train, y_train)
print("Model training completed.")
# Prediction
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print(f"Accuracy : {acc:.4f}")

    "accuracy": float(accuracy)
}

with open("model/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Model metrics saved.")


# Save Model
joblib.dump(model, "model/model.pkl")

print("Model Saved Successfully")
