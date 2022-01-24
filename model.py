import wrangle as w
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from scipy import stats 

# importing functions and code from wrangle module
import wrangle as w

import matplotlib.pyplot as plt
import seaborn as sns

# Turn off pink warning boxes in notebook
import warnings
warnings.filterwarnings("ignore")

# Handle large numbers w/o using scientific notation
pd.options.display.float_format = '{:.3f}'.format

# Prevent df columns from being truncated
pd.set_option('display.max_columns', None) 

import sklearn.preprocessing
# feature selection
from sklearn.feature_selection import SelectKBest, f_regression, RFE 
# modeling
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
# evaluation
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

def prep_modeling():
    '''
    
    '''
    df = w.clean_data(w.merge_data(df_left = w.acquire_data('311_service_data.csv')\
                        , df_right = w.acquire_data('med_incomebyzip.csv')))

    # get dummies
    df = pd.concat([df.days_to_close, \
        pd.get_dummies(df[['sourceid', 'category', 'council_distr']])], axis = 1)
    
    # split data
    train_validate, test = train_test_split(df, test_size=.2, random_state=12)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=12)

    # X and y split
    X_train = train.drop(columns = ['days_to_close'])
    y_train = train.days_to_close
    y_train_df = pd.DataFrame(y_train)

    X_validate = validate.drop(columns = ['days_to_close'])
    y_validate = validate.days_to_close
    y_validate_df = pd.DataFrame(y_validate)

    X_test = test.drop(columns = ['days_to_close'])
    y_test = test.days_to_close
    y_test_df = pd.DataFrame(y_test)

    return X_train, y_train, y_train_df, X_validate, y_validate, y_validate_df, X_test, y_test, y_test_df




def model(X_train, y_train, y_train_df, X_validate, y_validate, y_validate_df, X_test, y_test, y_test_df):
    '''
    
    '''

    ### baseline
    mean = y_train.mean()
    y_train_df['baseline'] = mean

    # calculating baseline RMSE 
    RMSE_train_baseline = np.sqrt(mean_squared_error(y_train_df.days_to_close, y_train_df.baseline))
    print(f'Baseline RMSE | Train, In Sample: {RMSE_train_baseline}')
    print()

    ### model 1 | Linear Regression `lr` model
    # creating linear regression object
    lr = LinearRegression()

    # fitting the object to train
    # lr.fit(X_train, y_train_df.days_to_close)
    lr.fit(X_train, y_train)

    # # predict 
    y_train_df['lr_pred'] = lr.predict(X_train)
    y_validate_df['lr_pred'] = lr.predict(X_validate)

    # evalutating lr model prediction
    RMSE_train_lr = np.sqrt(mean_squared_error(y_train_df.days_to_close, y_train_df.lr_pred))
    RMSE_validate_lr = np.sqrt(mean_squared_error(y_validate_df.days_to_close, y_validate_df.lr_pred))

    print(f'LinearRegression RMSE | Train, In Sample: {RMSE_train_lr}')
    print(f'LinearRegression RMSE | Validate, Out of Sample: {RMSE_validate_lr}')
    print()

    ### model 2 | LassoLars 'lars' model
    # creating the lasso lars object
    lars = LassoLars(alpha = 2.0)

    # fitting the object to train
    lars.fit(X_train, y_train)

    # predict
    y_train_df['lars_pred'] = lars.predict(X_train)
    y_validate_df['lars_pred'] = lars.predict(X_validate)

    # evaluate
    RMSE_train_lars = np.sqrt(mean_squared_error(y_train_df.days_to_close, y_train_df.lars_pred))
    RMSE_validate_lars = np.sqrt(mean_squared_error(y_validate_df.days_to_close, y_validate_df.lars_pred))

    print(f'LassoLars RMSE | Train, In Sample: {RMSE_train_lars}')
    print(f'LassoLars RMSE | Validate, Out of Sample: {RMSE_validate_lars}')
    print()

    ### making predictions on test data using best performing model, model 1 | Linear Regression
    # predictions on test 
    y_test_df['lr_pred'] = lr.predict(X_test)

    # evaluate on test
    RMSE_test_lr = np.sqrt(mean_squared_error(y_test_df.days_to_close, y_test_df.lr_pred))
    print(f'LinearRegression RMSE | Test {RMSE_test_lr}')
    print(f'Difference between model prediction and baseline {RMSE_train_baseline - RMSE_test_lr}')
    print()
    print()

