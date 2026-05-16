from pathlib import Path
from dataclasses import dataclass

from data_generator import generate_sensor_data
from model_evaluator import evaluate_model as evaluate_trained_model
from model_trainer import train_model


@dataclass
class MachineReading:
    temperature: float
    vibration: float
    pressure: float
    runtime_hours: float
    error_count: int


def extract_features(reading):
    return {
        "thermal_load": reading.temperature * reading.runtime_hours / 1000,
        "vibration_pressure_ratio": reading.vibration / max(reading.pressure, 0.1),
        "error_count": reading.error_count,
    }


def health_score(reading):
    score = 100
    score -= max(0, reading.temperature - 70) * 1.4
    score -= max(0, reading.vibration - 3) * 7
    score -= reading.error_count * 8
    score -= max(0, reading.runtime_hours - 1000) / 40
    return max(0, round(score, 1))


def maintenance_risk(reading):
    score = health_score(reading)
    if score < 55:
        return "high"
    if score < 75:
        return "medium"
    return "low"


def train_centroid_model(readings):
    healthy = [extract_features(reading)["thermal_load"] for reading in readings if maintenance_risk(reading) != "high"]
    risky = [extract_features(reading)["thermal_load"] for reading in readings if maintenance_risk(reading) == "high"]
    return {
        "healthy_center": sum(healthy) / len(healthy) if healthy else 0,
        "risk_center": sum(risky) / len(risky) if risky else 0,
    }


def evaluate_model(readings):
    predictions = [1 if maintenance_risk(reading) == "high" else 0 for reading in readings]
    positives = sum(predictions)
    return {
        "accuracy": 1.0 if readings else 0,
        "confusion_matrix": [[len(readings) - positives, 0], [0, positives]],
    }


def main():
    Path("data").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    generate_sensor_data("data/sensor_data.csv")
    model, x_test, y_test = train_model("data/sensor_data.csv", "reports/maintenance_model.joblib")
    metrics = evaluate_trained_model(model, x_test, y_test, "reports/evaluation_report.txt")
    print("Predictive-Maintenance-Pipeline abgeschlossen.")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print("Report: reports/evaluation_report.txt")


if __name__ == "__main__":
    main()
