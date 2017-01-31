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

path_train = '/home/slou/Documents/M2/semestre1/ML_project/facial_features_train.csv'
path_test = '/home/slou/Documents/M2/semestre1/ML_project/facial_features_test.csv'
#path_dataframe_train = '/Users/domitillecoulomb/Documents/DATA_SCIENCE/MachineLearning/Projet/portrait/dataframe.csv'
path_dataframe_train = '/home/slou/Documents/M2/semestre1/ML_project/portrait/dataframes/dataframe.csv'
path_dataframe_test = '/home/slou/Documents/M2/semestre1/ML_project/portrait/dataframes/dataframe_test.csv'

#Facial Features
df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)

#Extracted Features
df_extract_train = pd.read_csv(path_dataframe_train)
df_extract_train = df_extract_train.sort('ID')

df_extract_test = pd.read_csv(path_dataframe_test)
df_extract_test = df_extract_test.sort('ID')

#Final impact dataframes
df_impacts_train = pd.DataFrame(columns=('ID', 'position_n'))
df_impacts_train['ID'] = df_train['ID']
df_impacts_test = pd.DataFrame(columns=('ID','position_n'))
df_impacts_test['ID'] = df_test['ID']


"position impact"
x_train_position = df_extract_train[df_extract_train.columns[5:16]]
x_test_position = df_extract_test[df_extract_test.columns[5:16]]

y_train_position_p = df_train['position_impact_p']
y_train_position_n = df_train['position_impact_n']

# Positive
reg_position_p, a = mi.position_impact(x_train_position, y_train_position_p)
y_pred_position_p = reg_position_p.predict(x_test_position)
df_impacts_train['position_p'] = a
df_impacts_test['position_p']=y_pred_position_p
                
#Negative
reg_position_n, a  = mi.position_impact(x_train_position, y_train_position_n)
y_pred_position_n = reg_position_n.predict(x_test_position)
df_impacts_train['position_n'] = a
df_impacts_test['position_n']=y_pred_position_n

                

""" Angle impact """
x_train_angle = mi.angle_features(df_train)
x_test_angle = mi.angle_features(df_test)

y_train_angle_n = df_train["angle_impact_n"]
y_train_angle_p = df_train["angle_impact_p"]

# Positive
reg_angle_p, a = mi.angle_impact(x_train_angle, y_train_angle_p)
y_pred_angle_p = reg_angle_p.predict(x_test_angle)
df_impacts_train['angle_p'] = a
df_impacts_test['angle_p']=y_pred_angle_p


#Negative
reg_angle_n, a = mi.angle_impact(x_train_angle, y_train_angle_n)
y_pred_angle_n = reg_angle_n.predict(x_test_angle)
df_impacts_train['angle_n'] = a
df_impacts_test['angle_n']=y_pred_angle_n

                

"""expression impact"""

X_test = np.array(df_test[df_test.columns[6:75].append(df_test.columns[84:94])])

# Positive
reg_exp_p, a = mi.expression_p_svr(df_train)
y_pred_exp_p = reg_exp_p.predict(X_test)
df_impacts_train['expression_p'] = a
df_impacts_test['expression_p']=y_pred_exp_p

#Negative
reg_exp_n, a = mi.expression_n_svr(df_train)
y_pred_exp_n = reg_exp_n.predict(X_test)
df_impacts_train['expression_n'] = a
df_impacts_test['expression_n']=y_pred_exp_n
                
                

"""sharpness impact"""

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
df_impacts_train['sharpness_p'] = a
df_impacts_test['sharpness_p']=y_pred_sharp_p

#Negative
Y_n_train = np.array(df_train['sharpness_impact_n'])
reg_sharp_n, a,b = mi.sharpness_svr(X_scale_train, Y_n_train)
y_pred_sharp_n = reg_sharp_n.predict(X_scale_test)
df_impacts_train['sharpness_n'] = a
df_impacts_test['sharpness_n']=y_pred_sharp_n

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
df_impacts_train['background_p'] = a
df_impacts_test['background_p']=y_pred_back_p

#Negative background
Y_n_train = np.array(df_train['background_impact_n'])
reg_back_n, a,b = mi.background(X_scale_train, Y_n_train)
y_pred_back_n = reg_back_n.predict(X_scale_test)
df_impacts_train['background_n'] = a
df_impacts_test['background_n']=y_pred_back_n

#Positive Exposure
Y_p_train = np.array(df_train['exposure_impact_p'])
reg_expo_p, a,b = mi.background(X_scale_train, Y_p_train)
y_pred_expo_p = reg_expo_p.predict(X_scale_test)
df_impacts_train['exposure_p'] = a
df_impacts_test['exposure_p']=y_pred_expo_p

#Negative Exposure
Y_n_train = np.array(df_train['exposure_impact_n'])
reg_expo_n, a,b = mi.background(X_scale_train, Y_n_train)
y_pred_expo_n = reg_expo_n.predict(X_scale_test)
df_impacts_train['exposure_n'] = a
df_impacts_test['exposure_n']=y_pred_expo_n


"Results"
#y_pred_expo_n , y_pred_expo_p, y_pred_back_n , y_pred_back_p
# y_pred_sharp_n, y_pred_sharp_p, y_pred_exp_n, y_pred_exp_p


df_impacts_train.to_csv('/home/slou/Documents/M2/semestre1/ML_project/essai_impact.csv')
df_impacts_test.to_csv('/home/slou/Documents/M2/semestre1/ML_project/essai_impact_test.csv')

