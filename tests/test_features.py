import pandas as pd

from feature_engineering import add_engineered_features, split_features_and_target


def test_add_engineered_features_creates_expected_columns():
    frame = pd.DataFrame({
        "temperature": [80.0],
        "vibration": [4.0],
        "pressure": [2.5],
        "runtime_hours": [1200],
        "error_count": [2],
        "maintenance_needed": [1],
    })
    result = add_engineered_features(frame)
    assert "temp_vib_ratio" in result.columns
    assert "runtime_error_factor" in result.columns


def test_split_features_and_target_returns_matching_lengths():
    frame = pd.DataFrame({
        "temperature": [70.0, 90.0],
        "vibration": [2.0, 7.0],
        "pressure": [2.1, 3.5],
        "runtime_hours": [300, 3400],
        "error_count": [0, 5],
        "maintenance_needed": [0, 1],
    })
    features, target = split_features_and_target(frame)
    assert len(features) == len(target)
