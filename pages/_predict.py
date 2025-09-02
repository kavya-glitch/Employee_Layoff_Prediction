import pickle
import os
import pandas as pd

# --------------------------
# 1️⃣ Locate the model file
# --------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "layoff_model.pkl")

# --------------------------
# 2️⃣ Load the trained model
# --------------------------
with open(model_path, "rb") as f:
    model = pickle.load(f)

print("✅ Model loaded successfully!\n")

# --------------------------
# 3️⃣ Take employee input dynamically
# --------------------------
age = int(input("Age: "))
department = int(input("Department (numeric code from training): "))
jobrole = int(input("JobRole (numeric code from training): "))
years = int(input("Years at Company: "))
salary = int(input("Salary: "))
score = int(input("Performance Score: "))

sample_employee = pd.DataFrame([{
    "Age": age,
    "Department": department,
    "JobRole": jobrole,
    "YearsAtCompany": years,
    "Salary": salary,
    "PerformanceScore": score
}])

# --------------------------
# 4️⃣ Make prediction
# --------------------------
prediction = model.predict(sample_employee)[0]
probability = model.predict_proba(sample_employee)[0][1]

# --------------------------
# 5️⃣ Show results
# --------------------------
print("\nPrediction (0 = Safe, 1 = Risk of Layoff):", prediction)
print(f"Layoff Probability: {probability:.2f}")




