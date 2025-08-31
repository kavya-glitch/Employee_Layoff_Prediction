import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Load or Train the Model
# -----------------------------
# If you want, you can save and load the model from previous step:
# model.save("layoff_model.h5")
# model = load_model("layoff_model.h5")

# For simplicity, we retrain the model here (or import from preprocess_and_train)
from preprocess_and_train import X_train_rnn, y_train, X_test_rnn, y_test, model, scaler

st.title("Employee Layoff Prediction")

st.write("Enter employee details:")

# -----------------------------
# Input Form
# -----------------------------
age = st.number_input("Age", 18, 65, 30)
department = st.selectbox("Department", ["HR", "Sales", "Tech", "Finance"])
jobrole = st.selectbox("Job Role", ["Manager", "Executive", "Staff"])
years = st.number_input("Years at Company", 0, 40, 5)
salary = st.number_input("Salary", 10000, 200000, 50000)
performance = st.number_input("Performance Score (1-5)", 1, 5, 3)

# -----------------------------
# Prepare Input for Model
# -----------------------------
# One-hot encoding manually
dept_map = ["Department_HR", "Department_Sales", "Department_Tech", "Department_Finance"]
role_map = ["JobRole_Executive", "JobRole_Manager", "JobRole_Staff"]

input_dict = dict.fromkeys(dept_map + role_map, 0)

# Set selected department and role to 1
input_dict[f"Department_{department}"] = 1
input_dict[f"JobRole_{jobrole}"] = 1

# Combine all features
input_features = [age, years, salary, performance] + list(input_dict.values())

# Scale features
input_scaled = scaler.transform([input_features])

# Reshape for RNN
input_rnn = input_scaled.reshape((1, 1, len(input_scaled[0])))

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Layoff"):
    prediction = model.predict(input_rnn)[0][0]
    if prediction > 0.5:
        st.error(f"⚠️ High risk of layoff ({prediction*100:.2f}%)")
    else:
        st.success(f"✅ Low risk of layoff ({prediction*100:.2f}%)")
