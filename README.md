# autogluon-wrapper

The goals of this project are to create wrappers for the classes of AutoGluon such as TabularPredictor to achieve compatibility with the classifiers and the regressors of scikit-learn.

**!!!FUTUREWARNING!!!**  
**autogluon-wrapper will be renamed to Scikit-AutoGluon when released version 2.0 on 8/14/2021. The reason for the name change is to help you better understand the purpose of this project.**

## installing

To install autogluon-wrapper, execute the following command in your python environment.

```bash
pip install git+https://github.com/kzkymn/autogluon-wrapper.git#egg=autogluon-wrapper
```

**!!!FUTUREWARNING!!!**  
**At the next version 0.2, agwrapper will be renamed skautogluon.**

## how to use

If you have the following code that uses AutoGluon as follows,

```python
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
label = 'class'
save_path = 'agModels-predictClass'
# This train_data is a DataFrame which has columns of independent variables and a column named 'class' which means the objective variable.
predictor = TabularPredictor(label=label, path=save_path).fit(train_data)
```

You can write the equivalent code using autogluon-wrapper as if you use a scikit-learn predictor. Like this,

```python
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
X_train = train_data.drop(columns='class')
y_train = train_data['class']

save_path = 'agModels-predictClass'
predictor = TabularPredictorWrapper(path=save_path).fit(X_train, y_train)
```

So with autogluon-wrapper, you can incorporate AutoGluon's learning logics into your codes that are supposed to work with the scikit-learn frameworks, such as Pipeline.
