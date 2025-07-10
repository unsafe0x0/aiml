import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
data = pd.read_csv("dataset.csv")
data = pd.get_dummies(data, columns=["ocean_proximity"])

data.head()

# Separate features and target variable
X = data.drop(columns=["median_house_value"])
y = data["median_house_value"]

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Generate polynomial features
poly = PolynomialFeatures(degree=2)
X = pd.DataFrame(poly.fit_transform(X), columns=poly.get_feature_names_out(X.columns))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Results
# Mean Squared Error: 4428711320.981657
# R-squared: 0.6620359449210441