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

path_fft = '/Users/domitillecoulomb/Documents/DATA_SCIENCE/MachineLearning/Projet/facial_features_train.csv'
path_dataframe = '/Users/domitillecoulomb/Documents/DATA_SCIENCE/MachineLearning/Projet/portrait/dataframe.csv'


"expression impact"
df = pd.read_csv(path_fft)

reg_exp_p, a,b = mi.expression_p_svr(df)
reg_exp_n, a,b = mi.expression_n_svr(df)

"sharpness impact"
df_extract = pd.read_csv(path_dataframe)
df_extract = df_extract.sort('ID')

X_scale = preprocessing.scale(
                np.array(
                    df_extract[df_extract.columns[3:5]
                            .append(df_extract.columns[13:14])
                            .append(df_extract.columns[18:20])
                            .append(df_extract.columns[21:29])]))

Y_p = np.array(df['sharpness_impact_p'])
reg_sharp_p, a,b = mi.sharpness_svr(X_scale, Y_p)

Y_n = np.array(df['sharpness_impact_n'])
reg_sharp_n, a,b = mi.sharpness_svr(X_scale, Y_n)


