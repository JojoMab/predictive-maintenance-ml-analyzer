import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from feature_engineering import split_features_and_target


def train_model(data_path="data/sensor_data.csv", model_path="reports/maintenance_model.joblib"):
    frame = pd.read_csv(data_path)
    features, target = split_features_and_target(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    model = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
    model.fit(x_train, y_train)
    joblib.dump(model, model_path)
    return model, x_test, y_test
