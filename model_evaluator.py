from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_model(model, x_test, y_test, report_path="reports/evaluation_report.txt"):
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    lines = [
        "Predictive Maintenance Evaluation Report",
        "========================================",
        "",
        f"Accuracy: {metrics['accuracy']:.3f}",
        f"Precision: {metrics['precision']:.3f}",
        f"Recall: {metrics['recall']:.3f}",
        f"F1: {metrics['f1']:.3f}",
        f"Confusion Matrix: {metrics['confusion_matrix']}",
    ]
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return metrics
