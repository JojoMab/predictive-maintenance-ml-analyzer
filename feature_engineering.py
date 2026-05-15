import pandas as pd


FEATURE_COLUMNS = [
    "temperature",
    "vibration",
    "pressure",
    "runtime_hours",
    "error_count",
    "temp_vib_ratio",
    "runtime_error_factor",
]

TARGET_COLUMN = "maintenance_needed"


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create simple engineered features for the maintenance model."""
    engineered = data.copy()
    engineered["temp_vib_ratio"] = engineered["temperature"] / engineered["vibration"].replace(0, 0.1)
    engineered["runtime_error_factor"] = engineered["runtime_hours"] * (engineered["error_count"] + 1)
    return engineered


def split_features_and_target(data: pd.DataFrame):
    """Return feature matrix X and target vector y."""
    engineered = add_engineered_features(data)
    missing_columns = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in engineered.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return engineered[FEATURE_COLUMNS], engineered[TARGET_COLUMN]
