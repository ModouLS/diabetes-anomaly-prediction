import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import timedelta

# --- Config ---
st.set_page_config(page_title="Diabetes Dashboard", layout="wide")
st.title("🧠 Diabetes Glucose Prediction & Anomaly Detection")

# --- Load data and model ---
df = pd.read_csv("../models/559-ws-training_with_anomalies.csv", parse_dates=["timestamp"])
model = joblib.load("../models/glucose_rf_model.pkl")

# --- Predict glucose 30 min ahead ---
features = ['glucose', 'glucose_change', 'glucose_rolling_mean', 'hour', 'dayofweek']
df["glucose_pred"] = model.predict(df[features])

# --- Plot section ---
st.subheader("📈 Glucose with Anomalies and Predictions")

fig, ax = plt.subplots(figsize=(14, 5))

# Plot glucose values
ax.plot(df["timestamp"], df["glucose"], label="Glucose", color="blue")

# Anomalies
anomalies = df[df["anomaly"] == True]
ax.scatter(anomalies["timestamp"], anomalies["glucose"], color="red", label="Anomaly", s=25)

# Predictions
ax.plot(df["timestamp"], df["glucose_pred"], label="Predicted Glucose (30min ahead)", linestyle="--", color="green")

ax.set_xlabel("Time")
ax.set_ylabel("Glucose (mg/dL)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- Show data preview ---
st.subheader("📋 Data Snapshot")
st.dataframe(df.tail(20))
