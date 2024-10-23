import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

# Load the dataset
boston = fetch_openml(name='boston', version=1, as_frame=True)
data = boston.frame

# Exploratory Data Analysis (EDA)
print(data.info())
print(data.describe())

# Visualizing the distribution of the target variable
sns.histplot(data['MEDV'], bins=30, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('House Price')
plt.ylabel('Frequency')
plt.show()

# Visualizing relationships
sns.pairplot(data, x_vars=['CRIM', 'RM', 'AGE', 'LSTAT'], y_vars='MEDV', height=5)
plt.show()

# Correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Data Preprocessing
# Handling missing values (if any)
data = data.dropna()

# Normalizing/Standardizing features
features = data.drop('MEDV', axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(features_scaled, data['MEDV'], test_size=0.2, random_state=42)

# Model Implementation
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    results[name] = {'MAE': mae, 'RMSE': rmse}

# Model Evaluation
results_df = pd.DataFrame(results).T
print(results_df)
