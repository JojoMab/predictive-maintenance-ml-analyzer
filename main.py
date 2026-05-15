import pandas as pd

from data_generator import DATA_PATH, generate_sensor_data
from model_evaluator import evaluate_model
from model_trainer import train_model


def run_pipeline() -> dict:
    generate_sensor_data(DATA_PATH)
    data = pd.read_csv(DATA_PATH)
    model, x_test, y_test = train_model(data)
    metrics = evaluate_model(model, x_test, y_test)
    return metrics


if __name__ == "__main__":
    result = run_pipeline()
    print("Predictive maintenance pipeline completed.")
    print("Accuracy:", round(result["accuracy"], 3))
    print("Precision:", round(result["precision"], 3))
    print("Recall:", round(result["recall"], 3))
    print("Report: reports/evaluation_report.txt")
