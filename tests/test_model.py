from pathlib import Path

from data_generator import generate_sensor_data
from model_evaluator import evaluate_model
from model_trainer import train_model


def test_model_is_trainable_and_reaches_basic_accuracy(tmp_path):
    data_path = tmp_path / "sensor_data.csv"
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "evaluation_report.txt"
    generate_sensor_data(data_path, rows=600, random_state=7)
    model, x_test, y_test = train_model(data_path, model_path)
    metrics = evaluate_model(model, x_test, y_test, report_path)
    assert model_path.exists()
    assert report_path.exists()
    assert metrics["accuracy"] > 0.70
