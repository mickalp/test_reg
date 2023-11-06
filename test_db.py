#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:26:10 2023

@author: michal
"""

import pandas as pd
import sqlite3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Load data from an SQLite database using SQL
conn = sqlite3.connect('example_dataset.db')
query = "SELECT * FROM your_regression_data_table"
df = pd.read_sql_query(query, conn)
conn.close()

# Step 2: Data Preprocessing
X = df[['feature1', 'feature2', 'feature3']]
y = df['target']

# Additional Preprocessing Steps
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Build and train a regression model
# Create a dictionary of regression models to try
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': XGBRegressor()
}

best_model = None
best_score = -float('inf')

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    if r2 > best_score:
        best_score = r2
        best_model = model_name

# Step 5: Make predictions with the best model
best_model = models[best_model]
y_pred = best_model.predict(X_test)

# Step 6: Evaluate the best model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 7: Data Visualization
sns.scatterplot(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs. Predicted Values')
plt.show()

# Step 8: Cross-Validation
cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='r2')
print("Cross-Validation R-squared scores:", cv_scores)

# Step 9: Hyperparameter Tuning (Example for Random Forest)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='r2')
grid_search.fit(X_scaled, y)

best_rf_model = grid_search.best_estimator_
print("Best Random Forest Model:", best_rf_model)

# Step 10: PyTorch Neural Network
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = X_train.shape[1]
model = NeuralNet(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer.step()

X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_pred_torch = model(X_test_torch).detach().numpy()

# Step 11: Evaluate the PyTorch model
mse_torch = mean_squared_error(y_test, y_pred_torch)
r2_torch = r2_score(y_test, y_pred_torch)

# Step 12: Compare Results
print("Mean Squared Error (Best Model):", mse)
print("R-squared (Best Model):", r2)
print("Mean Squared Error (PyTorch):", mse_torch)
print("R-squared (PyTorch):", r2_torch)
