#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 13:28:38 2017

@author: domitillecoulomb
"""
import model_impacts as mi
import numpy as np
import pandas as pd
from sklearn import preprocessing


""" Getting the data  """

path_train = '/Users/domitillecoulomb/Documents/DATA_SCIENCE/MachineLearning/Projet/facial_features_train.csv'
path_test = ''
path_dataframe_train = '/Users/domitillecoulomb/Documents/DATA_SCIENCE/MachineLearning/Projet/portrait/dataframe.csv'
path_dataframe_test = ''

#Facial Features
df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)

#Extracted Features
df_extract_train = pd.read_csv(path_dataframe_train)
df_extract_train = df_extract_train.sort('ID')

df_extract_test = pd.read_csv(path_dataframe_test)
df_extract_test = df_extract_test.sort('ID')

"expression impact"

X_test = np.array(df_test[df_test.columns[6:75].append(df_test.columns[84:94])])

# Positive
reg_exp_p, a,b = mi.expression_p_svr(df_train)
y_pred_exp_p = reg_exp_p.predict(X_test)

#Negative
reg_exp_n, a,b = mi.expression_n_svr(df_train)
y_pred_exp_n = reg_exp_n.predict(X_test)

"sharpness impact"

X_scale_train = preprocessing.scale(
                np.array(
                    df_extract_train[df_extract_train.columns[3:5]
                            .append(df_extract_train.columns[13:14])
                            .append(df_extract_train.columns[18:20])
                            .append(df_extract_train.columns[21:29])]))

X_scale_test = preprocessing.scale(
                np.array(
                    df_extract_test[df_extract_test.columns[3:5]
                            .append(df_extract_test.columns[13:14])
                            .append(df_extract_test.columns[18:20])
                            .append(df_extract_test.columns[21:29])]))
#Positive
Y_p_train = np.array(df_train['sharpness_impact_p'])
reg_sharp_p, a,b = mi.sharpness_svr(X_scale_train, Y_p_train)
y_pred_sharp_p = reg_sharp_p.predict(X_scale_test)

#Negative
Y_n_train = np.array(df_train['sharpness_impact_n'])
reg_sharp_n, a,b = mi.sharpness_svr(X_scale_train, Y_n_train)
y_pred_sharp_n = reg_sharp_n.predict(X_scale_test)

"background impact and Exposure"
"Same Model"

X_scale_train = preprocessing.scale(
                np.array(
                    df_extract_train[df_extract_train.columns[2:]]))

X_scale_test = preprocessing.scale(
                np.array(
                    df_extract_test[df_extract_test.columns[2:]]))

#Positive background
Y_p_train = np.array(df_train['background_impact_p'])
reg_back_p, a,b = mi.background(X_scale_train, Y_p_train)
y_pred_back_p = reg_back_p.predict(X_scale_test)

#Negative background
Y_n_train = np.array(df_train['background_impact_n'])
reg_back_n, a,b = mi.background(X_scale_train, Y_n_train)
y_pred_back_n = reg_back_n.predict(X_scale_test)

#Positive Exposure
Y_p_train = np.array(df_train['exposure_impact_p'])
reg_expo_p, a,b = mi.background(X_scale_train, Y_p_train)
y_pred_expo_p = reg_expo_p.predict(X_scale_test)

#Negative Exposure
Y_n_train = np.array(df_train['exposure_impact_n'])
reg_expo_n, a,b = mi.background(X_scale_train, Y_n_train)
y_pred_expo_n = reg_expo_n.predict(X_scale_test)


"Results"
#y_pred_expo_n , y_pred_expo_p, y_pred_back_n , y_pred_back_p
# y_pred_sharp_n, y_pred_sharp_p, y_pred_exp_n, y_pred_exp_p


