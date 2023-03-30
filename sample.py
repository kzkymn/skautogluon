# %%
# Import required libraries
from skautogluon import TabularPredictorWrapper
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Generate sample data
X = np.random.rand(100, 1) * 10
y = 3 * X**2 + 2*X + 3 + np.random.randn(100, 1)

X = pd.DataFrame(X)
y = pd.Series(y.squeeze())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create a linear regression model
predictor = TabularPredictorWrapper()

# Fit the model to the training data
predictor.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = predictor.predict(X_test)

# %%
# Calculate some evaluation metrics(e.g. RMSE, MSE, MAE)
# The reason why the signs of values such as RMSE are flipped in the output is
# that the evaluation metrics calculated by AutoGluon are
# standardized so that the larger the value, the higher the accuracy.
predictor.evaluate(X_test, y_test, silent=True)

# %%
# List the models created by AutoGluon in a table in order of accuracy.
predictor.leaderboard(X_test, y_test, silent=True)

# %%
