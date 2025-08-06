# 🧠 Glucose Anomaly Detection & Prediction App

Welcome to the **Glucose Anomaly Detection & Prediction** dashboard — a Streamlit-based web application for visualizing blood glucose data, detecting anomalies, and predicting future glucose levels using machine learning.

---

## 🚀 What This App Does

- 📊 Visualizes continuous glucose monitoring (CGM) data
- 🔍 Automatically detects anomalies using Isolation Forest
- 🔮 Predicts glucose levels 30 minutes ahead using a trained Random Forest model
- 📌 Highlights hypoglycemic events (<70 mg/dL)

---

## 📁 Files Included

- `app.py`: Main Streamlit dashboard
- `models/559-ws-training_with_anomalies.csv`: Preprocessed glucose data with anomaly labels
- `models/glucose_rf_model.pkl`: Trained Random Forest model for glucose forecasting
- `requirements.txt`: Python dependencies

---

## ▶️ How to Use

1. The app runs automatically after deployment.
2. Glucose values and predicted future values are displayed in an interactive plot.
3. Anomalies and hypoglycemia events are clearly marked in red.

---

## 🧠 Model & Data

- **Dataset**: OhioT1DM KaggleHub CGM XML (converted to CSV)
- **Anomaly Detection**: Isolation Forest (scikit-learn)
- **Prediction Model**: Random Forest Regressor trained to predict glucose 30 minutes ahead

---

## 👤 Author

**Modou Singhateh**  
Physicist & Machine Learning Enthusiast  
📍 Based in Heidelberg, Germany

---

## 📬 Questions?

Feel free to reach out via Hugging Face or contribute improvements via pull request. Let's make diabetes monitoring smarter!
