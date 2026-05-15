import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MachineReading:
    temperature: float
    vibration: float
    pressure: float
    runtime_hours: float
    failure: int


def load_readings(path="data/machine_readings.csv"):
    with open(path, newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        return [
            MachineReading(
                temperature=float(row["temperature"]),
                vibration=float(row["vibration"]),
                pressure=float(row["pressure"]),
                runtime_hours=float(row["runtime_hours"]),
                failure=int(row["failure"]),
            )
            for row in rows
        ]


def extract_features(reading):
    return {
        "thermal_load": max(0, reading.temperature - 65),
        "vibration_load": reading.vibration,
        "pressure_delta": abs(reading.pressure - 2.2),
        "runtime_load": reading.runtime_hours / 1000,
    }


def health_score(reading):
    features = extract_features(reading)
    score = 100
    score -= features["thermal_load"] * 1.4
    score -= features["vibration_load"] * 7.5
    score -= features["pressure_delta"] * 8
    score -= max(0, reading.runtime_hours - 1000) / 75
    return max(0, round(score, 1))


def maintenance_risk(reading):
    score = health_score(reading)
    if score < 55:
        return "high"
    if score < 75:
        return "medium"
    return "low"


def train_centroid_model(readings):
    grouped = {0: [], 1: []}
    for reading in readings:
        grouped[reading.failure].append(health_score(reading))

    all_scores = grouped[0] + grouped[1]
    fallback_center = sum(all_scores) / len(all_scores)
    return {
        "healthy_center": sum(grouped[0]) / len(grouped[0]) if grouped[0] else fallback_center,
        "failure_center": sum(grouped[1]) / len(grouped[1]) if grouped[1] else fallback_center,
    }


def predict_failure(reading, model):
    score = health_score(reading)
    distance_to_healthy = abs(score - model["healthy_center"])
    distance_to_failure = abs(score - model["failure_center"])
    return 1 if distance_to_failure <= distance_to_healthy else 0


def evaluate_model(readings):
    sorted_readings = sorted(readings, key=lambda reading: reading.failure)
    split_index = max(2, int(len(sorted_readings) * 0.7))
    train_set = sorted_readings[:split_index]
    test_set = sorted_readings[split_index:] or sorted_readings
    model = train_centroid_model(train_set)

    predictions = [predict_failure(reading, model) for reading in test_set]
    actual = [reading.failure for reading in test_set]
    true_positive = sum(1 for predicted, real in zip(predictions, actual) if predicted == 1 and real == 1)
    true_negative = sum(1 for predicted, real in zip(predictions, actual) if predicted == 0 and real == 0)
    false_positive = sum(1 for predicted, real in zip(predictions, actual) if predicted == 1 and real == 0)
    false_negative = sum(1 for predicted, real in zip(predictions, actual) if predicted == 0 and real == 1)

    accuracy = (true_positive + true_negative) / len(test_set)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0

    return {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "confusion_matrix": {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
    }


def create_report(readings, output_path="reports/maintenance_report.txt"):
    evaluation = evaluate_model(readings)
    lines = [
        "Predictive Maintenance ML Report",
        "================================",
        "",
        f"Readings analyzed: {len(readings)}",
        f"Accuracy: {evaluation['accuracy']}",
        f"Precision: {evaluation['precision']}",
        f"Recall: {evaluation['recall']}",
        f"Confusion matrix: {evaluation['confusion_matrix']}",
        "",
        "Sample risks",
    ]
    for reading in readings:
        lines.append(
            f"temperature={reading.temperature} vibration={reading.vibration} "
            f"health={health_score(reading)} risk={maintenance_risk(reading)}"
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    readings = load_readings()
    report_path = create_report(readings)
    evaluation = evaluate_model(readings)
    print("Report generated successfully.")
    print(f"Accuracy: {evaluation['accuracy']}")
    print(f"Precision: {evaluation['precision']}")
    print(f"Recall: {evaluation['recall']}")
    print(f"Report: {report_path}")
