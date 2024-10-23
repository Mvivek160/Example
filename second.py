import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = {
    'CustomerID': range(1, 101),
    'Age': np.random.randint(18, 70, 100),
    'Tenure': np.random.randint(1, 10, 100),
    'Churn': np.random.choice([0, 1], 100) 
}

df = pd.DataFrame(data)


average_age_tenure = df.groupby('Churn').agg({'Age': 'mean', 'Tenure': 'mean'}).reset_index()
print("Average Age and Tenure by Churn Status:")
print(average_age_tenure)


X = df[['Age', 'Tenure']]
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy:.2f}")
