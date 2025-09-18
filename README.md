# AI-Powered Real-Time Health Monitoring

## Summary
Realtime IoT + ML demo that monitors heart rate, SpO₂ and steps and raises alerts when vitals are anomalous.

## Tech stack
Python, pandas, scikit-learn (IsolationForest), Streamlit, Plotly.

## Run locally
1. python -m venv .venv && activate it
2. pip install -r requirements.txt
3. python src/data_gen.py
4. python src/train_model.py
5. streamlit run src/app.py

## What to highlight in an interview
- Real-time simulation + model inference
- Feature engineering and anomaly detection tradeoffs
- How you’d replace the simulator with a real Kafka stream or an edge device (Raspberry Pi + TFLite)

## Future work
- LSTM/Autoencoder for sequence anomalies
- Kafka + Docker for production streaming
- Authentication, logging, HIPAA controls

## Dashboard
[Click Here](https://real-time-health-monitoring.streamlit.app/) to view the streamlit deployment
