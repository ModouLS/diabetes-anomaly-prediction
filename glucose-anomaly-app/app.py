import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Glucose Anomaly App", layout="wide")
st.title("📊 Glucose Anomaly Detection & Prediction")

# Load data and model
df = pd.read_csv("models/559-ws-training_with_anomalies.csv", parse_dates=["timestamp"])
model = joblib.load("models/glucose_rf_model.pkl")

# Predict glucose 30min ahead
features = ['glucose', 'glucose_change', 'glucose_rolling_mean', 'hour', 'dayofweek']
df["glucose_pred"] = model.predict(df[features])

# Plot
st.subheader("📈 Glucose with Anomalies and Prediction")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["timestamp"], df["glucose"], label="Glucose")
ax.scatter(df[df["anomaly"] == 1]["timestamp"], df[df["anomaly"] == 1]["glucose"], color='red', label="Anomaly", s=25)
ax.plot(df["timestamp"], df["glucose_pred"], label="Predicted (30min)", linestyle="--")
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("Glucose (mg/dL)")
ax.grid(True)
st.pyplot(fig)

# Data preview
st.subheader("📋 Data Snapshot")
st.dataframe(df.tail(20))
