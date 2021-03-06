#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 14:11:13 2018

@author: larrylaw
"""
"""
RESULTS:
min_mae: 947.9 with imputer w/ extension and OHE
best_leaf_node: ?
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer

def impute(data):
    """
    Input: Independent Variable
    Output: Imputed data with cols. Excluded columns with objects
    """
    #converting all int columns to int64 type
    data_int = data.select_dtypes(exclude = 'object')
    data_columns = data_int.columns
    data_imputer = Imputer()
    data_int = pd.DataFrame(data_imputer.fit_transform(data_int), columns = data_columns)
    return data_int

def impute_extension(data):
    """
    Input: Independent Variable
    Output: Imputed data with cols with missing. Excluded columns with objects
    """
    data_imputed = impute(data)
    data = data.select_dtypes(exclude = 'object')
    cols_with_missing = (col for col in data if data[col].isnull().any())
    for col in cols_with_missing:
        data_imputed[col + "was missing"] = data[col].isnull()
    return data_imputed

def OHE(data):
    """
    Input: Independent Variable
    Output: OHE data
    """
    #test getting dummy var
    # how to settle na for OHE?
    # Which cat to reject for OHE?
    # align both data sets
    data_object = data.select_dtypes(include = 'object')
    data_dummies =  pd.get_dummies(data_object, dummy_na = True)
    return data_dummies

def get_best_leaf_nodes(train_X, test_X, train_y, test_y):
    """
    Input:
        train_X/y: training data 
        test_X/y: testing data
    Output: int, Best Leaf Node
    """
    min_mae = train_y.iloc[0]
    best_leaf_nodes = 0
    for elt in range(2,1000,50):
        mae = 0
        mae = get_mae(train_X, test_X, train_y, test_y, max_leaf_nodes = elt)
        if mae < min_mae:
            min_mae = mae
            best_leaf_nodes = elt
    return best_leaf_nodes

def get_mae(train_X, test_X, train_y, test_y, max_leaf_nodes = None):
    """
    Input: 
        train_X/y: training data
        test_X/y: testing data 
        max_leaf_nodes: int, optional for RandomTreeRegressor. Consider get_best_leaf_nodes
    Output: MAE
    """
    model = RandomForestRegressor(max_leaf_nodes = max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    mae = mean_absolute_error(test_y, pred_y)
    return mae

def test_csv(true_test, train_X, test_X, train_y, file_name, max_leaf_nodes = None):
    """
    Input:
        true_test: Original test data. To obtain test ID
        X: Use entire of training data (not the train_test_split data)
        y: Testing data
        file_name: string, name of submission
        leaf_node: int, optional. Consider get_best_leaf_node function
    Output: Generates predicted results of test in csv format
    """
    model = RandomForestRegressor(max_leaf_nodes = max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y) #insert all data for better prediction values
    test_y = model.predict(test_X)
    test_df = pd.DataFrame({'Id': true_test.Id, 'SalePrice': test_y})
    return test_df.to_csv(file_name, index = False)

#Reading Data
iowa_data = pd.read_csv("Iowa Housing Prices.csv") 

#Setting iv and dv
data_imputed = impute_extension(iowa_data)
data_OHE = OHE(iowa_data)
X = data_imputed.join(data_OHE)
y = iowa_data.SalePrice

#train_test_split
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y,random_state = 0)

#Align training and test data
train_X, test_X = train_X.align(test_X, join="inner", axis=1)

#testing get_mae function
mae = get_mae(train_X, test_X, train_y, test_y)
best_leaf_nodes = get_best_leaf_nodes(train_X, test_X, train_y, test_y)

#Applying on Test data
#true_test = pd.read_csv("test.csv")
#true_data_imputed = impute_extension(true_test)
#true_data_OHE = OHE(true_test)
#true_test_X = true_data_imputed.join(true_data_OHE)
#X, true_test_X = X.align(true_test_X, join="inner", axis = 1)
#test_csv(true_test, X, true_test_X, y, "iowa_submission2.csv", max_leaf_nodes = best_leaf_nodes)


"""Archive
# Evaluating MAE
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
    
# get predicted prices on validation data
val_predictions = iowa_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

#Random Forest    
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
iowa_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, iowa_preds))

#Get Min Mae
def get_min_mae(pred_train, pred_val, targ_train, targ_val):
    min_mae = targ_train.iloc[0]
    for max_leaf_nodes in range(1,1000,10):
        mae = 0
        model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
        model.fit(pred_train, targ_train)
        preds_val = model.predict(pred_val)
        mae = mean_absolute_error(targ_val, preds_val)
        if mae < min_mae:
            min_mae = mae
    return min_mae
    
#Predicting test set
iowa_model = RandomForestRegressor(max_leaf_nodes=best_leaf_node)
iowa_model.fit(X, y) #inserted all data for better prediction values
iowa_preds = iowa_model.predict(test_X)
iowa_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': iowa_preds})
iowa_submission.to_csv('iowa_submission.csv', index = False)

#Get min mae and best leaf node
min_mae, best_leaf_node = get_min_mae_and_best_leaf_node(train_X, val_X, train_y, val_y)
"""
