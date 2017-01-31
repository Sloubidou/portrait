#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 13:53:39 2017

@author: domitillecoulomb
"""
import numpy as np
import pandas as pd
#from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


""" Position Impact """

def position_impact(X_train, y_train):
    
    
    reg = RandomForestRegressor(random_state=0, n_estimators=1000)
    reg.fit(X_train, y_train)
    
    y_pred = reg.predict(X_train)    
    print("Mean Squared error: {}", mean_squared_error(y_train, y_pred))
    
    return reg, y_pred


""" Angle Impact """
def angle_features(df):
    x_angle = df.ix[:,'pan angle':'roll angle']

    x_angle['middle_x'] = (df['x0'] + df['width'])/2
    x_angle['middle_y'] = (df['y0'] + df['height'])/2
    x_angle['nose_tip_x'] = df['nose_tip_x']
    x_angle['nose_tip_y'] = df['nose_tip_y']
    x_angle['nose_left_eye_dist'] = abs(df['nose_tip_x'] - df['left_eye_pupil_x'])
    x_angle['nose_right_eye_dist'] = abs(df['nose_tip_x'] - df['right_eye_pupil_x'])
    x_angle['mouth_eye'] = abs(df['mouth_center_x'] - df['midpoint_between_eyes_x'])
    
    return x_angle
    
def angle_impact(X_train, y_train):
    
    reg = RandomForestRegressor(random_state=0, n_estimators=1000)
    reg.fit(X_train, y_train)
    
    y_pred = reg.predict(X_train)    
    print("Mean Squared error: {}", mean_squared_error(y_train, y_pred))
    
    return reg, y_pred

"""
Regression Model for expression impacts
"""

def expression_p_svr(df): 
    
    "SVR - Best Result"
    
    X_train = df[df.columns[6:75].append(df.columns[84:94])]
    y_train = df['expression impact_p']

    reg = SVR(kernel='poly', C=0.8, gamma=0.1)
    reg.fit(X_train, y_train)
    #Prediction
    y_pred_train = reg.predict(X_train)
    print("Mean Squared error: {}", mean_squared_error(np.array(y_train), np.array(y_pred_train)))
    return reg, y_pred_train

    
def expression_n_svr(df): 
    
    "SVR - Best Result"
    
    X_train = df[df.columns[6:75].append(df.columns[84:94])]
    y_train = df['expression_impact_n']

    reg = SVR(kernel='poly', C=0.8, gamma=0.1)
    reg.fit(X_train, y_train)
    #Prediction
    y_pred_train = reg.predict(X_train)
    print("Mean Squared error: {}", mean_squared_error(np.array(y_train), np.array(y_pred_train)))
    return reg, y_pred_train

    
"""
Classification Model for expression impacts
"""

def expression_p_svc(df):
    
    "SVC"
    
    classes = np.sort(df['expression impact_p'].unique())
    labels = np.arange(24)

    df_clf = df['expression impact_p'].copy()
    df_clf = df_clf.replace(classes, labels)
    
    X_train, X_test, y_train, y_test = train_test_split(df[df.columns[6:75].append(df.columns[84:94])], df_clf, train_size=0.8, random_state=0)
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
    clf.fit(X_train, y_train)

    #Prediction
    y_pred = clf.predict(X_test)
    y_pred_clf = pd.DataFrame(y_pred).replace(labels, classes)
    y_test_clf = pd.DataFrame(y_test).replace(labels, classes)
    print("Mean Squared error: {}", mean_squared_error(np.array(y_test_clf), np.array(y_pred_clf)))
    return clf, y_pred, y_test

""" 
Sharpness Impact 
"""

#df_extract = pd.read_csv(path_dataframe)
#df_extract = df_extract.sort('ID')

#X_scale = preprocessing.scale(
#                np.array(
#                    df_extract[df_extract.columns[3:5]
#                            .append(df_extract.columns[13:14])
#                            .append(df_extract.columns[18:20])
#                            .append(df_extract.columns[21:29])]))

#Y_p = np.array(df['sharpness_impact_p'])
#Y_n = np.array(df['sharpness_impact_n'])

def sharpness_svr(X_scale, Y):
    
    
    reg = SVR(kernel='rbf', C=0.8, gamma=0.1)
    #reg = LinearRegression()
    reg.fit(X_scale, Y)

    #Prediction
    y_train_pred = reg.predict(X_scale)
    print("Mean Squared error: {}", mean_squared_error(Y, y_train_pred))
    return reg, y_train_pred, Y
 
    
""" 
Background and Exposure Impacts
""" 
    
def background(X_scale, Y):
    
    reg = SVR(kernel='rbf', C=0.8, gamma=0.1)
    reg.fit(X_scale, Y)

    #Prediction
    y_train_pred = reg.predict(X_scale)
    print("Mean Squared error: {}", mean_squared_error(Y, y_train_pred))
  
    
    return reg, y_train_pred, Y


