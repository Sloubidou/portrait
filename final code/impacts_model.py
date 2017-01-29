#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 13:53:39 2017

@author: domitillecoulomb
"""
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
import pandas as pd

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
    

