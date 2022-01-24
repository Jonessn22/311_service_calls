'''
MODULE OF FUNCTIONS AND CODE USED TO WRANGLE PROJECT DATA AND PREPARE FOR EXPLORATION
'''

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


def acquire_data(file_name):
    '''
THIS FUNCTION TAKES IN A CSV FILE_NAMES AND READS THE DATA, RETURNING A PANDAS DF.

INPUTS
    - FILENAME: (STR)
OUTPUTS
    - DATAFRAME: (PD DF)
    '''

    df = pd.read_csv(file_name)
    print(f'{file_name} read complete.')
    print(f'There are {df.shape[0]} rows and {df.shape[1]} columns in {file_name}.')
    print()

    return df

def merge_data(df_left, df_right):
    '''
    
INPUTS:
    - DF_LEFT: (PD DF)
    - DF_RIGHT: (PD DF)
    
    '''

    '''
for loop that will loop through each address value in df_left and add last 5 chars to list
    '''
    # creating empty list to hold zip codes for df_left
    zips = []

    # looping through each col and adding the last 5 characters of the address to the zips list
    for address in df_left.OBJECTDESC:
        zips.append(address[-5:])
        
    # creating a new zip column for the df with those values
    df_left['zip'] = zips
        
    # dropping the OBJECTDESC address column from the df
    df_left.drop(columns = ['OBJECTDESC'], inplace = True)
    print('Left DF Zip Column Ready for Merge.')

    '''
changing column dtype and creating for loop that will change dtype for zip codes in df_right 
from numeric to string
    '''

    # changing col dtype from int to object
    df_right['Zip Code'] = df_right['Zip Code'].astype('O')
    
    # creating empty list to hold zip codes for df_right
    Zip_Code = []
    
    # updating Zip Code col in df_right to list of strings
    for value in df_right['Zip Code']:
        Zip_Code.append(str(value))

    # updating Zip Code column series to new, list of strings in df_right
    df_right['Zip Code'] = Zip_Code

    print('Right DF Zip Column Ready for Merge.')

    '''
merging df_left and df_right
    '''
    df = df_left.merge(df_right, how = 'inner', left_on = 'zip', right_on = 'Zip Code')\
        .drop(columns = ['Zip Code'])
    print('DFs Merged.')
    print(f'There are {df.shape[0]} rows and {df.shape[1]} columns in the merged df.')
    print()

    return df

def clean_data(df):
    '''
    
    '''
    # initial column drop
    cols_to_drop = ['#', 'TYPENAME', 'XCOORD', 'YCOORD', 'Report Starting Date', 'Report Ending Date']
    df.drop(columns = cols_to_drop, inplace = True)
    
    # removing empty spaces from column names and lowercasing
    df.columns = df.columns.str.replace(' ', '').str.lower()

    # updating columns names
    cols_to_rename = {'caseid':'case_ref_num', 
                        'openeddatetime':'case_open', 
                        'sla_date':'sla_due', 
                        'closeddatetime':'case_close', 
                        'late(yes/no)':'sla_late', 
                        'reasonname':'dept_div',  
                        'casestatus':'case_status', 
                        'councildistrict':'council_distr', 
                        'avg_household_income':'avg_inc'}
    df.rename(columns = cols_to_rename, inplace = True)

    # changing column dtypes and dropping rows with null dates 
    # to object
    df[['case_ref_num', 'council_distr']] = df[['case_ref_num', 'council_distr']].astype('O')
    
    # dropping rows with nulls
    df.dropna(inplace = True)

    # to datetime
    df.sla_due = pd.to_datetime(df.sla_due)
    df.case_open = pd.to_datetime(df.case_open)
    df.case_close = pd.to_datetime(df.case_close)

    # population from object to integer
    df.population = df.population.str.replace(',', '').astype(int)

    # to float
    df.avg_inc = df.avg_inc.str.replace('$', '').str.replace(',', '').astype(float)

    # dropping location column
    df.drop(columns =  ['location'], inplace = True)

    # feature engineering new column: 
    df['days_to_close'] = (df.case_close - df.case_open).astype('str')\
        .str.replace(' days', '').astype(int)

    # resetting index and dropping old noncontinuous index columns
    df.reset_index(inplace = True)
    df.drop(columns = ['index'], inplace = True)

    # dropping case_close column since we are only looking at cases that have been closed 
    # dropping case_ref_num column since unqiue for each row
    # dropping case status since only has one value
    df.drop(columns = ['case_close', 'case_ref_num', 'case_status'], inplace = True)
    
    # plots distributions of each numeric column
    df.hist(figsize = (20, 10))
    plt.suptitle('Distributions of Numeric Variable Values')
    plt.show()
    
    # column dtype lists
    # creating empty lists to be appended in for loop
    obj_list = []
    int_list = []
    float_list = []
    bool_list = []
    date_list = []

    # for loop to append df columns to corresponding lists
    for col in df.columns:
        if df[col].nunique() == 2:
            bool_list.append(col)
            
        elif df[col].dtype == 'O':
            obj_list.append(col)
            
        elif df[col].dtype == 'int':
            int_list.append(col)

        elif df[col].dtype == 'float':
            float_list.append(col)
        
        else:
            date_list.append(col)
        
    print(f'Object List:\n{obj_list}\n\nInteger List:\n{int_list}\n\nBool List:\n{bool_list}\n\nFloat List:\n{float_list}\n\nDates List:\n{date_list}')
    print()
    print(f'There are {df.shape[0]} rows and {df.shape[1]} columns in the cleaned df.')
    print('DataFrame is cleaned and ready for splitting.')

    return df

def split_data(df):
    '''
    '''

    # creating test dataset
    train_validate, test = train_test_split(df, test_size=.2, random_state=12)

    # creating the train and test datasets
    train, validate = train_test_split(train_validate, test_size=.3, random_state=12)

    # verifying the split
    print('Data has been split.')
    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')
    print()

    return train, validate, test
