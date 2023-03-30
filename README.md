# skautogluon(Scikit-AutoGluon)

Scikit-AutoGluon project aims to provide a scikit-learn compatible API for AutoGluon's TabularPredictor.

## installing

To install Scikit-AutoGluon, first set up AutoGluon according to the following URL.  
[https://auto.gluon.ai/stable/install.html](https://auto.gluon.ai/stable/install.html)

Next, run the following command in a python environment (e.g. a conda environment with AutoGluon installed).

```bash
pip install git+https://github.com/kzkymn/skautogluon.git#egg=skautogluon
```

## how to use

Given the following code that uses AutoGluon,

```python
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
label = 'class'
save_path = 'agModels-predictClass'
# This train_data is a DataFrame which has columns of independent variables and a column named 'class' which means the objective variable.
predictor = TabularPredictor(label=label, path=save_path).fit(train_data)
```

You can write the equivalent code using skautogluon as if you use a scikit-learn predictor. Like this,

```python
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
X_train = train_data.drop(columns='class')
y_train = train_data['class']

save_path = 'agModels-predictClass'
predictor = TabularPredictorWrapper(path=save_path).fit(X_train, y_train)
```

With skautogluon, you can incorporate AutoGluon's learning logics into your codes that are supposed to work with the scikit-learn frameworks, such as Pipeline.

If you need an entire code implementation example, please refer to `sample.py` in this project.

## About Docker File Support

Since the AutoGluon official provides a Dockerfile, this project will not provide a Dockerfile.

For more details, please refer to the following link.

[https://hub.docker.com/r/autogluon/autogluon](https://hub.docker.com/r/autogluon/autogluon)