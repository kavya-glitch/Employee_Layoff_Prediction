
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



import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("employee_layoff_data.csv")
print("Original Dataset:")
print(df.head())

# -----------------------------
# 2. Encode Categorical Variables
# -----------------------------
df_encoded = pd.get_dummies(df, columns=['Department', 'JobRole'])
print("\nAfter Encoding:")
print(df_encoded.head())

# -----------------------------
# 3. Separate Features and Target
# -----------------------------
X = df_encoded.drop('Laid_Off', axis=1).values
y = df_encoded['Laid_Off'].values

# -----------------------------
# 4. Scale Features
# -----------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 5. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

# -----------------------------
# 1. Reshape Input for RNN
# -----------------------------
# RNN expects input as [samples, timesteps, features]
# We treat each employee as a sequence of features (timesteps=1)
X_train_rnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_rnn = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# -----------------------------
# 2. Build RNN Model
# -----------------------------
model = Sequential()
model.add(LSTM(32, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2]), activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# -----------------------------
# 3. Train the Model
# -----------------------------
history = model.fit(
    X_train_rnn, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test_rnn, y_test)
)

# -----------------------------
# 4. Evaluate Model
# -----------------------------
loss, accuracy = model.evaluate(X_test_rnn, y_test)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

