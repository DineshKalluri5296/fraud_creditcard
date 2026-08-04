import json

with open("model/training_stats.json") as f:
    TRAIN_STATS = json.load(f)


def calculate_drift(features):

    feature_names = list(TRAIN_STATS.keys())

    scores = []

    for name, value in zip(feature_names, features):

        mean = TRAIN_STATS[name]["mean"]

        std = TRAIN_STATS[name]["std"]

        z = abs(value - mean) / (std + 1e-8)

        scores.append(z)

    return round(sum(scores) / len(scores), 3)
