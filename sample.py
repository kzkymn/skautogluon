# %%
# Import required libraries
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset
from sklearn.model_selection import train_test_split

from skautogluon import TabularPredictorWrapper

# Create sample data
train_data = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")
X_train = train_data.drop(columns="class")
y_train = train_data["class"]

test_data = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")
X_test = test_data.drop(columns="class")
y_test = test_data["class"]

save_path = "agModels-predictClass"
predictor = TabularPredictorWrapper(path=save_path).fit(X_train, y_train)

# Fit the model to the training data
predictor.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = predictor.predict(X_test)

# %%
# Calculate some evaluation metrics(e.g. accuracy, f1, roc_auc)
predictor.evaluate(X_test, y_test, silent=True)

# %%
# List the models created by AutoGluon in a table in order of accuracy.
predictor.leaderboard(X_test, y_test, silent=True)

# %%
