import pandas as pd

from feature_engineering import add_engineered_features, split_features_and_target


def test_add_engineered_features_adds_expected_columns():
    data = pd.DataFrame(
        {
            "temperature": [80.0],
            "vibration": [4.0],
            "pressure": [7.0],
            "runtime_hours": [1200],
            "error_count": [2],
            "maintenance_needed": [1],
        }
    )

    result = add_engineered_features(data)

    assert "temp_vib_ratio" in result.columns
    assert "runtime_error_factor" in result.columns
    assert result.loc[0, "temp_vib_ratio"] == 20.0


def test_split_features_and_target_returns_matching_lengths():
    data = pd.DataFrame(
        {
            "temperature": [70.0, 90.0],
            "vibration": [3.0, 6.0],
            "pressure": [7.5, 6.0],
            "runtime_hours": [500, 7000],
            "error_count": [0, 4],
            "maintenance_needed": [0, 1],
        }
    )

    x, y = split_features_and_target(data)

    assert len(x) == 2
    assert len(y) == 2
    assert "temp_vib_ratio" in x.columns
