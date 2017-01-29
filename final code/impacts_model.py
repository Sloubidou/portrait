#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 13:53:39 2017

@author: domitillecoulomb
"""
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
import pandas as pd

""" Position Impact """

def position_impact(x_selection_position, y_position_impact):
    print("Calcul de position_impact :")
    X_train, X_test, y_train, y_test = train_test_split(x_selection_position, y_position_impact, train_size=0.8, random_state=0)
    print("Nb d'échantillons d'apprentissage :  {}".format(X_train.shape[0]))
    print("Nb d'échantillons de validation :    {}".format(X_test.shape[0]))

    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
                   
    svr_rbf.fit(X_train, y_train)
    y_pred = svr_rbf.predict(X_test)
    
    #mesure standard de performance
    from sklearn.metrics import mean_squared_error
    print("Accuracy       : ", mean_squared_error(y_test, y_pred))
    return y_pred

""" Expression Impact """ 

def expression_svr(df): 
    
    "SVR - Best Result"
    
    X_train, X_test, y_train, y_test = train_test_split(df[df.columns[6:75].append(df.columns[84:94])], df['expression impact_p'], train_size=0.8, random_state=0)
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    reg = SVR(kernel='poly', C=0.8, gamma=0.1)
    reg.fit(X_train, y_train)

    #Prediction
    y_pred = reg.predict(X_test)
    print("Mean Squared error: {}", mean_squared_error(np.array(y_test), np.array(y_pred)))
    return y_pred, y_test

def expression_svc(df):
    
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
    return y_pred, y_test


