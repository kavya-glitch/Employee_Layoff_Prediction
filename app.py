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


