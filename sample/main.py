# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from agwrapper import TabularPredictorWrapper
import sklearn.datasets
from sklearn.model_selection import train_test_split

# %%
# load datasets same as train.csv and test.csv in https://autogluon.s3.amazonaws.com/datasets/Inc/
X, y = sklearn.datasets.fetch_openml(data_id=179, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# %%
save_path = 'agModels-predictClass'
predictor = TabularPredictorWrapper(path=save_path).fit(X_train, y_train)

# %%
results = predictor.fit_summary()

# %%
print("AutoGluon infers problem type is: ", predictor.problem_type)
print("AutoGluon identified the following types of features:")
print(predictor.feature_metadata)

# %%
# unnecessary, just demonstrates how to load previously-trained predictor from file
predictor = TabularPredictorWrapper.load(save_path)

y_pred = predictor.predict(X_test)
print("Predictions:  \n", y_pred)
perf = predictor.evaluate_predictions(
    y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

# %%
predictor.leaderboard(X_test, y_test, silent=True)

# %%
predictor.predict(X_test, model='LightGBM')

# %%
time_limit = 60
metric = 'roc_auc'
predictor = TabularPredictorWrapper(eval_metric=metric)
predictor.fit(X_train, y_train, time_limit=time_limit, presets='best_quality')
predictor.leaderboard(X_test, y_test, silent=True)

# %%
