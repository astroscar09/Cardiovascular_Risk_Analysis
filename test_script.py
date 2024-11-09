import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colors import *
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix


def read_csv(file):
    return pd.read_csv(file)

def print_columns(df):
    
    num_cols = df.shape[1]

    print(f'Printing out the names of {num_cols} columns in the file:')
    print('-------------------------------------------------------------------')
    for col in df.columns:
        print(col)
    print()

def add_category_column(df, column):

    df[f'Cat_{column}'] = pd.factorize(copied_df[column])[0]
    
    return df

def add_binary_column(df, column):

    new_values = []
    for values in df[column].values:
        if values == 'Yes':
            new_values.append(1)
        else:
            new_values.append(0)
    df[f'Cat_{column}'] = new_values
    return df

def making_numerical_df(df):

    for column in df.columns.values:
        #this checks if it is a string and if so we pass the column into the add category variable function
        if df[column].dtype == 'O': 
            df = add_category_column(df, column)
    return df

#reading in the file
cardio_df = pd.read_csv('../data/CVD_cleaned.csv')

#making a copy of the file so as to not alter the original dataset
copied_df = cardio_df.copy()


#grabbing the features of the dataset
features = ['General_Health', 'Checkup', 'Exercise', 'Skin_Cancer', 'Other_Cancer', 
            'Depression', 'Diabetes', 'Arthritis', 'Sex', 'Age_Category', 'Height_(cm)', 
            'Weight_(kg)', 'BMI', 'Smoking_History', 'Alcohol_Consumption', 'Fruit_Consumption', 
            'Green_Vegetables_Consumption', 'FriedPotato_Consumption']

def grabbing_useful_features(df):

    features = ['Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption', 'Fruit_Consumption', 
            'Green_Vegetables_Consumption', 'FriedPotato_Consumption']

    features.extend([x for x in df.columns.values if 'Cat_' in x])

    reduced_df = df[features]

    return reduced_df

def converting_floats_to_ints(df, column):

    df[column] = df[column].values.astype(int)

    return df

def change_floats_to_int(df):

    features = ['Alcohol_Consumption', 'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption']

    for f in features:
        df[f] = df[f].values.astype(int)

    return df

def data_processing(file):

    df = read_csv(file)
    
    df = add_category_column(df, 'General_Health')
    df = add_category_column(df, 'Checkup')
    df = add_binary_column(df, 'Exercise')
    df = add_binary_column(df, 'Heart_Disease')
    df = add_binary_column(df, 'Skin_Cancer')
    df = add_binary_column(df, 'Other_Cancer')
    df = add_binary_column(df, 'Depression')
    df = add_binary_column(df, 'Diabetes')
    df = add_binary_column(df, 'Arthritis')
    df = add_category_column(df, 'Sex')
    df = add_category_column(df, 'Age_Category')
    df = add_binary_column(df, 'Smoking_History')
    df = change_floats_to_int(df)

    final_df = grabbing_useful_features(df)

    return final_df, df

predictor = 'Heart_Disease'


def perform_grid_search_on_hyperparameters(param_grid, model, X_train, y_train,
                                            search = 'Grid', **kwargs):
    
    if search == 'Grid':

        hyper_search = GridSearchCV(estimator=model, 
                                    param_grid=param_grid, 
                                    cv=5, 
                                    n_jobs=-1, 
                                    verbose=2)

    elif search == 'Random':
        hyper_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                          n_iter=100, cv=5, n_jobs=-1, verbose=2, 
                                          random_state=42)
        
    elif search == 'Bayes':
        hyper_search = BayesSearchCV(estimator=model, search_spaces=param_dist, 
                                               n_iter=32, cv=5, n_jobs=-1, verbose=2,
                                               random_state=42)
        


    print(f'Using Grid: {search}:')
    start = time.time()
    hyper_search.fit(X_train, y_train)
    end = time.time()

    tot_time = end - start
    hr = int(tot_time/3600)
    minutes = int((tot_time%3600)/60)
    seconds =  tot_time%3600%60

    print(f'Total Time for grid search is: {hr} hr, {minutes} min, {seconds}s')
    print("Best parameters found: ", hyper_search.best_params_)
    print("Best cross-validation score: ", hyper_search.best_score_) 

    return hyper_search.best_params_, hyper_search.best_score_



