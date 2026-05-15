from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

REPORT_PATH = Path("reports/evaluation_report.txt")


def evaluate_model(model, x_test, y_test, report_path: Path = REPORT_PATH) -> dict:
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(create_report(metrics), encoding="utf-8")
    return metrics


def create_report(metrics: dict) -> str:
    matrix = metrics["confusion_matrix"]
    return (
        "Predictive Maintenance Evaluation Report\n"
        "========================================\n\n"
        f"Accuracy: {metrics['accuracy']:.3f}\n"
        f"Precision: {metrics['precision']:.3f}\n"
        f"Recall: {metrics['recall']:.3f}\n"
        f"F1 Score: {metrics['f1']:.3f}\n\n"
        "Confusion Matrix:\n"
        f"Class 0 correct: {matrix[0][0]} | Class 0 predicted as 1: {matrix[0][1]}\n"
        f"Class 1 predicted as 0: {matrix[1][0]} | Class 1 correct: {matrix[1][1]}\n"
    )
