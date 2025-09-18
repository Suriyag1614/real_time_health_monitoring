# src/data_gen.py
import pandas as pd
import numpy as np
import datetime
import os

def generate_sample(path="data/sample_data.csv", n=5000, freq_seconds=2, seed=42):
    np.random.seed(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    start = datetime.datetime.now()
    rows = []
    hr = 70.0
    spo2 = 98.0
    steps = 0

    for i in range(n):
        t = start + datetime.timedelta(seconds=i * freq_seconds)
        # small random walk for HR and SpO2
        hr += np.random.normal(0, 1.2)
        spo2 += np.random.normal(0, 0.15)
        # occasional anomalies
        if np.random.rand() < 0.006:
            hr += np.random.uniform(20, 45)   # tachy anomaly
            spo2 -= np.random.uniform(4, 10)  # low SpO2 anomaly
        # steps
        steps += int(np.random.poisson(0.6))
        rows.append({
            "timestamp": t.isoformat(),
            "heart_rate": max(30, round(hr, 1)),
            "spo2": round(min(100, spo2), 1),
            "steps": steps
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")

if __name__ == "__main__":
    generate_sample()
