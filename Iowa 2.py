#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 14:11:13 2018

@author: larrylaw
"""

import pandas as pd
import numpy as np

#MAE
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

def get_min_mae_and_best_leaf_node(predictors_train, predictors_val, targ_train, targ_val):
    min_mae = targ_train.iloc[0]
    best_leaf_node = 0
    for elt in range(2,1000,10):
        mae = 0
        model = RandomForestRegressor(max_leaf_nodes=elt, random_state=0)
        model.fit(predictors_train, targ_train)
        preds_val = model.predict(predictors_val)
        mae = mean_absolute_error(targ_val, preds_val)
        if mae < min_mae:
            min_mae = mae
            best_leaf_node = elt
    return min_mae, best_leaf_node

#Reading Data
iowa_data = pd.read_csv("Iowa Housing Prices.csv") 
test = pd.read_csv("test.csv")

#Setting iv and dv
iowa_predictors = ["LotArea", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
X = iowa_data[iowa_predictors]
y = iowa_data.SalePrice
test_X = test[iowa_predictors]

#train_test_split
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

#Get min mae and best leaf node
min_mae, best_leaf_node = get_min_mae_and_best_leaf_node(train_X, val_X, train_y, val_y)

#Predicting test set
iowa_model = RandomForestRegressor(max_leaf_nodes=best_leaf_node, random_state = 0)
iowa_model.fit(X, y) #inserted all data for better prediction values
iowa_preds = iowa_model.predict(test_X)
iowa_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': iowa_preds})
iowa_submission.to_csv('iowa_submission.csv', index = False)

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
def get_min_mae(predictors_train, predictors_val, targ_train, targ_val):
    min_mae = targ_train.iloc[0]
    for max_leaf_nodes in range(1,1000,10):
        mae = 0
        model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
        model.fit(predictors_train, targ_train)
        preds_val = model.predict(predictors_val)
        mae = mean_absolute_error(targ_val, preds_val)
        if mae < min_mae:
            min_mae = mae
    return min_mae
"""
