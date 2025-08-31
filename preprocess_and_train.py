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
