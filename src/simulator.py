# src/simulator.py
import pandas as pd
import time

def stream_from_csv(path="data/sample_data.csv", delay_seconds=0.5):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    for _, row in df.iterrows():
        yield {
            "timestamp": row["timestamp"],
            "heart_rate": float(row["heart_rate"]),
            "spo2": float(row["spo2"]),
            "steps": float(row["steps"])
        }
        time.sleep(delay_seconds)
