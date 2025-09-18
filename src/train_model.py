# src/train_model.py
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
from utils import featurize

def train(input_csv="data/sample_data.csv", out_path="models/if_model.joblib"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.read_csv(input_csv, parse_dates=["timestamp"])
    df = featurize(df)
    features = ['heart_rate', 'spo2', 'steps', 'hr_diff', 'hr_roll_mean_5', 'spo2_roll_mean_5']
    X = df[features].fillna(0).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    clf.fit(Xs)

    joblib.dump({"model": clf, "scaler": scaler, "features": features}, out_path)
    print("Saved model to", out_path)

if __name__ == "__main__":
    train()
