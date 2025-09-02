
# app.py

import streamlit as st
import pandas as pd
import pickle
import os

# --------------------------
# 1️⃣ Set paths
# --------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, "layoff_model.pkl")
le_dept_path = os.path.join(base_dir, "le_department.pkl")
le_job_path = os.path.join(base_dir, "le_jobrole.pkl")

# --------------------------
# 2️⃣ Load model and encoders
# --------------------------
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(le_dept_path, "rb") as f:
    le_department = pickle.load(f)

with open(le_job_path, "rb") as f:
    le_jobrole = pickle.load(f)

# --------------------------
# 3️⃣ Streamlit UI
# --------------------------
st.title("Employee Layoff Prediction")
st.write("Enter employee details to predict the risk of layoff:")

age = st.number_input("Age", min_value=18, max_value=70, value=30)
department = st.selectbox("Department", ["Sales", "HR", "IT", "Finance"])
jobrole = st.selectbox("Job Role", ["Staff", "Executive", "Manager", "Analyst"])
years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
salary = st.number_input("Salary", min_value=10000, max_value=200000, value=50000)
performance_score = st.number_input("Performance Score", min_value=1, max_value=5, value=3)

# --------------------------
# 4️⃣ Convert inputs into DataFrame
# --------------------------
input_data = pd.DataFrame([{
    "Age": age,
    "Department": le_department.transform([department])[0],
    "JobRole": le_jobrole.transform([jobrole])[0],
    "YearsAtCompany": years_at_company,
    "Salary": salary,
    "PerformanceScore": performance_score
}])

# --------------------------
# 5️⃣ Predict
# --------------------------
if st.button("Predict Layoff Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    if prediction == 0:
        st.success(f"Safe from layoff ✅ (Probability: {probability:.2f})")
    else:
        st.error(f"Risk of layoff ⚠️ (Probability: {probability:.2f})")



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
