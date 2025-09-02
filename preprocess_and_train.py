# preprocess_and_train.py

import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import pickle

# --------------------------
# 1️⃣ Load the dataset
# --------------------------
data = pd.read_csv("employee_layoff_data.csv")
# Replace infinities with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN
data.dropna(inplace=True)

# Optional: reset index after dropping
data.reset_index(drop=True, inplace=True)
# Fill numeric columns with median
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
print("Any infinite values:", np.isinf(data[num_cols]).any().any())

data[num_cols] = data[num_cols].fillna(data[num_cols].median())

# Fill categorical columns with mode
cat_cols = data.select_dtypes(include='object').columns
for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])


print("Columns in dataset:", data.columns)
print(data.head())
print(data.info())
print(data.isna().sum())
# Check for missing or invalid values
print("Missing values per column:\n", data.isnull().sum())



# --------------------------
# 2️⃣ Rename target column if needed
# --------------------------
if "Layoff" in data.columns:
    data.rename(columns={"Layoff": "Layoff"}, inplace=True)

print("Columns after renaming:", data.columns.tolist())

# --------------------------
# 3️⃣ Encode categorical features
# --------------------------
le_department = LabelEncoder()
data["Department"] = le_department.fit_transform(data["Department"])
le_jobrole = LabelEncoder()

data["JobRole"] = le_jobrole.fit_transform(data["JobRole"])
# --------------------------
# 4️⃣ Split features and target
# --------------------------
X = data.drop(columns=["Laid_Off"])
y = data["Laid_Off"]

# --------------------------
# 5️⃣ Split into training and test sets
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = MLPClassifier(hidden_layer_sizes=(50,30), max_iter=1000, random_state=42)
model.fit(X_train, y_train)
# --------------------------
# 6️⃣ Train the model
# --------------------------
model = MLPClassifier(hidden_layer_sizes=(50,30), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# --------------------------
# 7️⃣ Save the trained model
# --------------------------
with open("layoff_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save fitted encoders
with open("le_department.pkl", "wb") as f:
    pickle.dump(le_department, f)

with open("le_jobrole.pkl", "wb") as f:
    pickle.dump(le_jobrole, f)
print("Any infinite values:", np.isinf(data[num_cols]).any().any())
numeric_data = data.select_dtypes(include=[np.number])
print("✅ Model trained and saved successfully as layoff_model.pkl")


