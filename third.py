import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('path/to/winequality-red.csv', sep=';')  # Update the path accordingly

# Display the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Handling missing values (if any)
data.fillna(data.mean(), inplace=True)  # Impute with mean

# Example of string manipulation (if there were text columns)
# data['text_column'] = data['text_column'].str.lower().str.strip()

# Convert relevant data columns to NumPy arrays
features = data.drop('quality', axis=1).values  # Features
target = data['quality'].values  # Target variable

# Perform basic statistics
mean_features = np.mean(features, axis=0)
median_features = np.median(features, axis=0)

print(f"Mean of features: {mean_features}")
print(f"Median of features: {median_features}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")


