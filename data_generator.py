import numpy as np
import pandas as pd


def generate_sensor_data(output_path="data/sensor_data.csv", rows=800, random_state=42):
    rng = np.random.default_rng(random_state)
    temperature = rng.normal(68, 9, rows).clip(40, 105)
    vibration = rng.normal(3.2, 1.4, rows).clip(0.4, 9.5)
    pressure = rng.normal(2.4, 0.45, rows).clip(1.1, 4.1)
    runtime_hours = rng.integers(50, 4500, rows)
    error_count = rng.poisson(1.6, rows)

    risk_score = (
        (temperature - 60) * 0.035
        + vibration * 0.45
        + (pressure - 2.0) * 0.55
        + runtime_hours / 2800
        + error_count * 0.55
    )
    threshold = np.quantile(risk_score, 0.72)
    maintenance_needed = (risk_score >= threshold).astype(int)

    frame = pd.DataFrame({
        "temperature": temperature.round(2),
        "vibration": vibration.round(2),
        "pressure": pressure.round(2),
        "runtime_hours": runtime_hours,
        "error_count": error_count,
        "maintenance_needed": maintenance_needed,
    })
    frame.to_csv(output_path, index=False)
    return frame


if __name__ == "__main__":
    generate_sensor_data()
