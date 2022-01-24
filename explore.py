import wrangle as w

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats 

def wrangle_data(file_name1, file_name2):
    '''
 THIS FUNCTION CALLS THE WRANGLE FUNCTIONS, TAKING IN TWO CSV FILE NAMES AND RETURNS
 THE MERGED DATA, SPLIT INTO TRAIN, VALIDATE, AND TESTING DFS. 

 IT ALSO REMOVES THE NEGATIVE DAYS_TO_CLOSE VALUES FROM EACH DF.
    '''

    train, validate, test = w.split_data(w.clean_data(w.merge_data(df_left = w.acquire_data(file_name1)\
                                                               , df_right = w.acquire_data(file_name2))))

    # filtering out the negative days to close values from train, validate, and test
    train = train[train.days_to_close >= 0]
    validate = validate[validate.days_to_close >= 0]
    test = test[test.days_to_close >= 0]

    return train, validate, test

def continuous_heatmap(train):
    '''
 THIS FUNCTION PLOTS A HEATMAP OF THE CONTINUOUS VARIABLE CORRELATIONS WITH THE TARGET AND
 PRINTS THE RESULTS OF SPEARMANS R STATISTICAL TESTS FOR EACH OF THE TWO CONTINUOUS VARIABLES 
 WITH THE TARGET DAYS_TO_CLOSE.   
    '''

    # creating a list of continuous variables
    contin_vars = ['days_to_close', 'avg_inc', 'population']

    # plotting heatmap
    plt.figure(figsize = (12, 8))
    sns.heatmap(train[contin_vars].corr(), annot = True, cmap = "coolwarm", \
        center = 0, linewidths = 4, linecolor = 'white')
    plt.title('Weak Correltion Between Continuous Variables and Target')
    plt.show()

    r, p = stats.spearmanr(train['days_to_close'], train['avg_inc'])
    print('Looking at days_to_close and avg_inc')
    print(f'The r-value is: {round(r, 2)}')
    print(f'The p-value is: {p}\nThere is {round(p, 1)}% chance that we see these results by chance.')
    print('--------------------------')
    print()
    r, p = stats.spearmanr(train['days_to_close'], train['population'])
    print('Looking at days_to_close and population:')
    print(f'The r-value or correlation coefficient is: {round(r, 2)}')
    print(f'The p-value is: {p}\nThere is {round(p, 1)}% chance that we see these results by chance.')
    print('--------------------------')
    print()

def explore(train):
    '''
THIS FUNCTION TAKES IN THE TRAINING DATA AND PLOTS EXPLORATION VISUALIZATIONS. 
    '''

    sns.barplot(x = 'sla_late', y = 'days_to_close', data = train)
    plt.title('Most Reported Service Requests are Closing After SLA Due Date')
    plt.show()
    print()

    plt.figure(figsize = (20, 8))
    sns.countplot(data = train, x = 'category')
    plt.title('Solid Waste and Property Maintenance Have Most Reported Service Requests')
    plt.show()

    plt.figure(figsize = (18, 8))
    sns.countplot(data = train, x = 'sourceid')
    plt.title('Little Use of 311 Mobile App | Most Service Requests Coming from Web Portal')
    plt.show()

    plt.figure(figsize = (18, 8))
    sns.boxplot(data = train, x = 'sourceid', y = 'days_to_close', hue = 'sla_late')
    plt.title('A lot of 311 App Requests are not Meeting SLA')
    plt.show()

# def split_and_scale(train, validate, test):
#     '''
# THIS FUNCTION SPLITS TRAIN, VALIDATE, AND TEST FOR MODELING AND GETS DUMMIES FOR EACH
# FEATURE (THEY ARE ALL CATEGORICAL).
#     '''

#     X_cols = ['sourceid', 'category', 'council_distr']

#     X_train = train[X_cols]
#     X_train = pd.get_dummies(X_train)
#     y_train = train.days_to_close
#     y_train_df = pd.DataFrame(y_train)

#     X_validate = validate[X_cols]
#     X_validate = pd.get_dummies(X_validate)
#     y_validate = validate.days_to_close
#     y_validate_df = pd.DataFrame(y_validate)

#     X_test = test[X_cols]
#     X_test = pd.get_dummies(X_test)
#     y_test = test.days_to_close
#     y_test_df = pd.DataFrame(y_test)

#     return X_train, y_train, y_train_df, X_validate, y_validate, y_validate_df, X_test, y_test, y_test_df






