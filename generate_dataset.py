# generate_dataset.py

import pandas as pd
import numpy as np

# --------------------------
# 1️⃣ Set random seed for reproducibility
# --------------------------
np.random.seed(42)

# --------------------------
# 2️⃣ Number of employees
# --------------------------
num_low_risk = 200   # mostly safe employees
num_high_risk = 50   # high-risk employees

# --------------------------
# 3️⃣ Generate low-risk employees
# --------------------------
low_risk = pd.DataFrame({
    "Age": np.random.randint(25, 50, size=num_low_risk),
    "Department": np.random.choice(["Sales", "HR", "IT", "Finance"], size=num_low_risk),
    "JobRole": np.random.choice(["Staff", "Executive", "Manager", "Analyst"], size=num_low_risk),
    "YearsAtCompany": np.random.randint(2, 15, size=num_low_risk),
    "Salary": np.random.randint(40000, 120000, size=num_low_risk),
    "PerformanceScore": np.random.randint(3, 6, size=num_low_risk),
    "Layoff": 0  # safe
})

# --------------------------
# 4️⃣ Generate high-risk employees
# --------------------------
high_risk = pd.DataFrame({
    "Age": np.random.randint(40, 60, size=num_high_risk),
    "Department": np.random.choice(["Sales", "HR", "IT", "Finance"], size=num_high_risk),
    "JobRole": np.random.choice(["Staff", "Executive", "Manager", "Analyst"], size=num_high_risk),
    "YearsAtCompany": np.random.randint(0, 5, size=num_high_risk),
    "Salary": np.random.randint(15000, 50000, size=num_high_risk),
    "PerformanceScore": np.random.randint(1, 3, size=num_high_risk),
    "Layoff": 1  # high risk
})

# --------------------------
# 5️⃣ Combine datasets
# --------------------------
data = pd.concat([low_risk, high_risk], ignore_index=True)

# --------------------------
# 6️⃣ Shuffle the dataset
# --------------------------
data = data.sample(frac=1).reset_index(drop=True)

# --------------------------
# 7️⃣ Save to CSV
# --------------------------
data.to_csv("employee_layoffs_data.csv", index=False)
print("✅ Dataset generated successfully with shape:", data.shape)
print(data.head())
