# Machine Learning Project 1 on Higgs Boson Competition

`Team`: 4 - 1 = 3 ducks

`Teammates`: Yan Fu, Shengzhao Xia, Wan-Tzu Huang

This project is adapted from the Kaggle 2014 competition [Higgs Boson Machine Learning Challenge] (https://www.kaggle.com/c/Higgs-boson). The competition orgins from the famous Higgs Boson experiment and the competitors need to improve observation of Higgs Boson from background noise using machine learning methods. 

In this project, we are given train set data (including 250000 events, with an ID column, 30 feature columns, a weight column and a label column) and test set data (including 550000 events with an ID column and 30 feature columns). Only Numpy and visualization libraries are allowed. 

To reproduce our result one will need to:

1. Install `python` and `Numpy` on you computer. In order to reproduce perfectly, please make sure that the versions are `python3.6.5` and `Numpy1.14.3` respectively;
2. Download `train.csv` and `test.csv` from [kaggle](https://www.kaggle.com/c/epfml18-higgs), and save them in `/data` document;
3. Run the python scripy `run.py` and the prediction result `Predicion.csv` will be produced in `/data` automatically .

Below we will introduce files and functions in our repotory.

## Auxiliary modules

* ### `helpers.py`

Contain the given helper functions, including `load_csv_data` (used to load csv file) and `create_csv_submission` (used to create output files in csv format for Kaggle submission).

* ### `tools.py`
  
 * `DataFrame`: A class wraps Numpy methods and is used to replace the pandas.DataFrame. It includes methods of copy,  reset_index, drop, loc, value counts and so on.

And in tools.py we also include functions used for data processing.

  * `standardize`: Standardize data by subtracting data's mean and then divide by data's standard deviation.

  * `log_process`: Take logarithm of positive data to narrow data range.

  * `build_polynomial_features`: Build polynomial features by squaring, root-squaring and multiplying bewteen features. 

Besides, in tools.py we include the cross validation functions using `predict_labels`, `build_k_indices`, `cross_validation` and `cv_loop`.

## Regression models

* ### `implementations.py`

Contains 6 implemented model functions, including `least_squares_GD`, `least_squares_SGD`, `least_squares`, `ridge_regression`, `logistic_regression` and `reg_logistic_regression`.

In addition, we also include other function that are needed by the model functions (namely, cost functions `calculate_logi_loss`, `calculate_mse`, `calculate_mae` and `compute_loss`, logistic function `logistic` and (stochastic) gradient descent `batch_iter` and `compute_gradient`).

## Analysis document

* ### `AnalysisDocument.ipynb`

A file explains the process, evaluates the results and presents evidence using plots and tables, which includes all data preparation, feature generation as well as cross-validation steps we have used.

## Prediction

* ### `run.py`

A python scripy use trained model and test data to produce prediction result `Predicion.csv` in `/data`, which is the same as what we uploaded to Kaggle.
