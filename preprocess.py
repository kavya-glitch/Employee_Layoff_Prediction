import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import pickle

# Features and target
X = data[['Age', 'Department', 'JobRole', 'YearsAtCompany', 'Salary', 'PerformanceScore']].values
y = data['Laid_Off'].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for RNN [samples, timesteps, features]
X_train_rnn = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_rnn = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Build RNN model
model = Sequential()
model.add(SimpleRNN(10, input_shape=(1, X_train.shape[1]), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train_rnn, y_train, epochs=20, batch_size=8, validation_data=(X_test_rnn, y_test))

# Save the trained model
pickle.dump(model, open("rnn_model.pkl", "wb"))

print("RNN model trained and saved successfully!")

# Load your dataset
data = pd.read_csv("employee_layoff_data.csv")  # replace with your dataset file name
print(data.head())
# Encode Department
le_dept = LabelEncoder()
data['Department'] = le_dept.fit_transform(data['Department'])
pickle.dump(le_dept, open("le_dept.pkl", "wb"))

# Encode JobRole
le_role = LabelEncoder()
data['JobRole'] = le_role.fit_transform(data['JobRole'])
pickle.dump(le_role, open("le_role.pkl", "wb"))

print("LabelEncoders saved successfully!")