# src/utils.py
import pandas as pd

def featurize(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    df['heart_rate'] = df['heart_rate'].astype(float)
    df['spo2'] = df['spo2'].astype(float)
    df['steps'] = df['steps'].astype(float)

    # basic features
    df['hr_diff'] = df['heart_rate'].diff().fillna(0)
    df['hr_roll_mean_5'] = df['heart_rate'].rolling(5, min_periods=1).mean()
    df['spo2_roll_mean_5'] = df['spo2'].rolling(5, min_periods=1).mean()
    return df
