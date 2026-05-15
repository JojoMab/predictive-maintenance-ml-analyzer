from pathlib import Path
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import split_features_and_target


MODEL_PATH = Path("reports/maintenance_model.pkl")


def train_model(data: pd.DataFrame, model_path: Path = MODEL_PATH, seed: int = 42):
    """Train a simple logistic regression model for maintenance classification."""
    x, y = split_features_and_target(data)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    pipeline.fit(x_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(pipeline, file)

    return pipeline, x_test, y_test
