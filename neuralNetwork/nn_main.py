import pandas as pd
import random
import numpy as np
from micrograd.engine import Value
from micrograd.nn import MLP
from parse import data2df, transform_columns

# Read and transform data
df = data2df('../Employee_Attrition.csv')
df = transform_columns(df)

if df is None:
    raise ValueError("Data loading failed.")

n_train = int(0.8 * len(df))
train_data = df.iloc[:n_train]
test_data = df.iloc[n_train:]

# Extract inputs and outputs
X_train = train_data.drop('Attrition', axis=1)
y_train = train_data['Attrition'].astype(float).values

X_test = test_data.drop('Attrition', axis=1)
y_test = test_data['Attrition'].astype(float).values

# Identify numerical columns (excluding one-hot encoded columns)
numerical_columns = ['Age', 'DistanceFromHome', 'Education', 'HourlyRate',
                     'JobLevel', 'JobSatisfaction', 'NumCompaniesWorked',
                     'TotalWorkingYears', 'WorkLifeBalance', 'YearsAtCompany',
                     'YearsSinceLastPromotion']

# Get indices of numerical columns in X_train
numerical_indices = [X_train.columns.get_loc(col) for col in numerical_columns]

# Convert X_train and X_test to numpy arrays
X_train_np = X_train.values.astype(float)
X_test_np = X_test.values.astype(float)

# Normalize only numerical columns
mins = X_train_np[:, numerical_indices].min(axis=0)
maxs = X_train_np[:, numerical_indices].max(axis=0)

def normalize(X, indices, mins, maxs):
    X_norm = X.copy()
    X_norm[:, indices] = (X_norm[:, indices] - mins) / (maxs - mins + 1e-8)
    return X_norm

X_train_norm = normalize(X_train_np, numerical_indices, mins, maxs)
X_test_norm = normalize(X_test_np, numerical_indices, mins, maxs)

# Define the model
input_size = X_train_norm.shape[1]
model = MLP(input_size, [32, 16, 1])

# Define the loss function
def binary_cross_entropy(y_pred, y_true):
    loss = - (y_true * y_pred.log() + (1 - y_true) * (1 - y_pred).log())
    return loss

# Training loop
epochs = 100
learning_rate = 0.02

for epoch in range(epochs):
    total_loss = 0
    for x, y_true in zip(X_train_norm, y_train):
        model.zero_grad()                           # Reset gradients
        x_values = [Value(xi) for xi in x]          # Convert inputs to list of Value objects
        y_pred = model(x_values)                    # Forward pass
        y_pred = y_pred.sigmoid()                   # Apply sigmoid activation
        loss = binary_cross_entropy(y_pred, y_true) # Compute loss
        total_loss += loss.data
        loss.backward()                             # Backward pass
        for p in model.parameters():                # Update parameters
            p.data -= learning_rate * p.grad
    avg_loss = total_loss / len(X_train_norm)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")

# Evaluation on test data
correct = 0
for x, y_true in zip(X_test_norm, y_test):
    x_values = [Value(xi) for xi in x]
    y_pred = model(x_values)
    y_pred = y_pred.sigmoid()
    y_pred_value = y_pred.data
    y_pred_class = 1 if y_pred_value > 0.5 else 0
    if y_pred_class == y_true:
        correct += 1
accuracy = correct / len(X_test_norm)
print(f"Test Accuracy: {accuracy * 100:.2f}%")