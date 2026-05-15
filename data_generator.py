from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/sensor_data.csv")


def generate_sensor_data(output_path: Path = DATA_PATH, rows: int = 800, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic machine sensor data for a simple maintenance classification task."""
    rng = np.random.default_rng(seed)

    temperature = rng.normal(72, 11, rows).clip(35, 120)
    vibration = rng.normal(4.2, 1.4, rows).clip(0.5, 10)
    pressure = rng.normal(7.5, 1.1, rows).clip(3, 12)
    runtime_hours = rng.integers(100, 9000, rows)
    error_count = rng.poisson(1.4, rows).clip(0, 8)

    risk_signal = (
        (temperature > 84).astype(int)
        + (vibration > 5.2).astype(int)
        + (pressure < 6.4).astype(int)
        + (runtime_hours > 6500).astype(int)
        + (error_count >= 3).astype(int)
    )

    probability = 1 / (1 + np.exp(-(risk_signal - 2.4)))
    maintenance_needed = (rng.random(rows) < probability).astype(int)

    data = pd.DataFrame(
        {
            "temperature": temperature.round(2),
            "vibration": vibration.round(2),
            "pressure": pressure.round(2),
            "runtime_hours": runtime_hours,
            "error_count": error_count,
            "maintenance_needed": maintenance_needed,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return data


if __name__ == "__main__":
    generated = generate_sensor_data()
    positive_rate = generated["maintenance_needed"].mean()
    print(f"Generated {len(generated)} rows at {DATA_PATH}")
    print(f"Positive class rate: {positive_rate:.2%}")
