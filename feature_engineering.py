FEATURE_COLUMNS = [
    "temperature",
    "vibration",
    "pressure",
    "runtime_hours",
    "error_count",
    "temp_vib_ratio",
    "runtime_error_factor",
]


def add_engineered_features(frame):
    data = frame.copy()
    data["temp_vib_ratio"] = data["temperature"] / data["vibration"].clip(lower=0.1)
    data["runtime_error_factor"] = data["runtime_hours"] * (data["error_count"] + 1)
    return data


def split_features_and_target(frame):
    data = add_engineered_features(frame)
    return data[FEATURE_COLUMNS], data["maintenance_needed"]
