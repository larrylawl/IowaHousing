#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 13:25:07 2018

@author: larrylaw
"""
"""
Changes: Change algo for producing test prediction
rmse_log: 0.125 using XGBR
Improvement: Fitting 100% of data, generating score through cross validation
Using: XGBR
Top 5 Indicators (gain): OveralQual, GarageCars, BsmtQual_Ex
Remarks: Did not use ESR as it worsens results

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor, plot_importance

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

def to_csv(test, pred_y, file_name):
    """
    Input:
        test: Original test data. To obtain test ID
        pred_y: Predicted DV based on XGBR model
    Output: Converts pred_y to csv format
    """
    y_df = pd.DataFrame({'Id': test.Id, 'SalePrice': pred_y})
    return y_df.to_csv(file_name, index = False)

#Reading Data
iowa_data = pd.read_csv("Iowa Housing Prices.csv") 

#Setting iv and dv
X = iowa_data.drop(labels = ["Id", "SalePrice"], axis = 1)
X_imputed = impute_extension(X)
X_OHE = OHE(X)
X = X_imputed.join(X_OHE)
y = iowa_data.SalePrice

#Model
XGBR_model = XGBRegressor(learning_rate=0.05, n_estimators= 1000)

#Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(XGBR_model, X, y, scoring = "neg_mean_squared_log_error", cv = 5)
rmse_log_score = np.sqrt(scores.mean()*-1)

#Partial Dependence Plot
#plot_importance(XGBR_model.fit(X,y), importance_type = "gain", max_num_features = 10)

#Applying on Test data
#Data preprocessing
test = pd.read_csv("test.csv")
test_X_imputed = impute_extension(test)
test_X_OHE = OHE(test)
test_X = test_X_imputed.join(test_X_OHE)

#Align as OHE generates diff no. of columns
#Inner as it guarantees both has data > more accurate predictions
X, test_X = X.align(test_X, join="inner", axis = 1) 


XGBR_model.fit(X, y)
test_y_pred = XGBR_model.predict(test_X)
to_csv(test, test_y_pred, "iowa_submission3.csv")