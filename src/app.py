# src/app.py
import streamlit as st
import pandas as pd
import time
import joblib
import plotly.express as px
from simulator import stream_from_csv
from utils import featurize

@st.cache_resource
def load_model(path="models/if_model.joblib"):
    data = joblib.load(path)
    return data['model'], data['scaler'], data['features']

st.set_page_config(page_title="AI Health Monitor", layout="wide")
st.title("AI-Powered Real-Time Health Monitoring 🩺")

model, scaler, features = load_model()

st.sidebar.header("Stream Controls")
delay = st.sidebar.slider("Delay (s) between datapoints", 0.1, 2.0, 0.5)
start = st.sidebar.button("Start Stream")
stop = st.sidebar.button("Stop")

# session state toggles
if 'running' not in st.session_state:
    st.session_state.running = False
if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

placeholder_chart = st.empty()
placeholder_log = st.container()

# data buffer for plotting
if 'buffer' not in st.session_state:
    st.session_state.buffer = pd.DataFrame(columns=['timestamp','heart_rate','spo2','steps','anomaly'])

if st.session_state.running:
    stream = stream_from_csv(delay_seconds=delay)
    for item in stream:
        df_row = pd.DataFrame([item])
        df_row = featurize(df_row)
        X = df_row[features].fillna(0).values
        Xs = scaler.transform(X)
        pred = model.predict(Xs)   # -1 anomaly, 1 normal
        is_anom = (pred[0] == -1)

        df_row['anomaly'] = int(is_anom)
        st.session_state.buffer = pd.concat([st.session_state.buffer, df_row[['timestamp','heart_rate','spo2','steps','anomaly']]], ignore_index=True)
        st.session_state.buffer = st.session_state.buffer.tail(200)

        # Plot heart rate with anomalies flagged
        fig = px.line(st.session_state.buffer, x='timestamp', y='heart_rate', title="Heart Rate (latest 200)")
        anom_pts = st.session_state.buffer[st.session_state.buffer['anomaly']==1]
        if not anom_pts.empty:
            fig.add_scatter(x=anom_pts['timestamp'], y=anom_pts['heart_rate'], mode='markers', marker=dict(size=10,symbol='x'), name='anomaly')
        placeholder_chart.plotly_chart(fig, use_container_width=True)

        # show small cards & logs
        latest = st.session_state.buffer.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Heart rate", f"{latest.heart_rate:.0f} bpm")
        c2.metric("SpO₂", f"{latest.spo2:.1f}%")
        c3.metric("Steps", f"{int(latest.steps)}")
        if is_anom:
            placeholder_log.warning(f"⚠️ Anomaly detected at {latest.timestamp}: HR={latest.heart_rate}, SpO₂={latest.spo2}")
        time.sleep(0.01)  # tiny pause to allow UI responsiveness
else:
    st.info("Press **Start Stream** in the sidebar to begin simulation.")
    # show last buffer if available
    if not st.session_state.buffer.empty:
        fig = px.line(st.session_state.buffer, x='timestamp', y='heart_rate', title="Heart Rate (last buffer)")
        placeholder_chart.plotly_chart(fig, use_container_width=True)
