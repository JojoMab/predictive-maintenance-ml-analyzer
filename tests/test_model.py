from data_generator import generate_sensor_data
from model_trainer import train_model
from model_evaluator import evaluate_model


def test_model_trains_and_reaches_basic_accuracy(tmp_path):
    data_path = tmp_path / "sensor_data.csv"
    model_path = tmp_path / "model.pkl"
    report_path = tmp_path / "report.txt"

    data = generate_sensor_data(output_path=data_path, rows=600, seed=7)
    model, x_test, y_test = train_model(data, model_path=model_path, seed=7)
    metrics = evaluate_model(model, x_test, y_test, report_path=report_path)

    assert metrics["accuracy"] > 0.70
    assert report_path.exists()
    assert model_path.exists()


def test_generated_data_contains_both_classes(tmp_path):
    data = generate_sensor_data(output_path=tmp_path / "sensor_data.csv", rows=600, seed=11)

    assert set(data["maintenance_needed"].unique()) == {0, 1}
    positive_rate = data["maintenance_needed"].mean()
    assert 0.20 <= positive_rate <= 0.40
