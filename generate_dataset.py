import pandas as pd
import numpy as np

# Number of sample employees
n = 500  # You can change this number

# For reproducibility (same random numbers every time)
np.random.seed(42)

# Generate random employee data
data = {
    'Age': np.random.randint(22, 60, n),
    'Department': np.random.choice(['HR', 'Sales', 'Tech', 'Finance'], n),
    'JobRole': np.random.choice(['Manager', 'Executive', 'Staff'], n),
    'YearsAtCompany': np.random.randint(0, 20, n),
    'Salary': np.random.randint(30000, 150000, n),
    'PerformanceScore': np.random.randint(1, 6, n),
    'Laid_Off': np.random.choice([0, 1], n, p=[0.8, 0.2])  # 20% layoffs
}

# Create a DataFrame
df = pd.DataFrame(data)

# Show the first 5 rows
print(df.head())

# Save the dataset to CSV
df.to_csv("employee_layoff_data.csv", index=False)

print("Dataset created and saved as employee_layoff_data.csv")
