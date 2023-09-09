import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as
import numpy as np

# Generate synthetic data
np.random.seed(0)
n_samples = 1000

# Generate features
feature1 = np.random.rand(n_samples)
feature2 = np.random.rand(n_samples)
# Add more features as needed

# Generate target variable (house prices)
price = 1000 + 200 * feature1 + 300 * feature2 + np.random.randn(n_samples) * 50
from sklearn.model_selection import train_test_split

X = np.column_stack((feature1, feature2))  # Features
y = price  # Target variable
X_train, X_test, y_train, y_test
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")