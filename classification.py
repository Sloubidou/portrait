#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:56:14 2017

@author: paulinenicolas
"""

import pandas as pd
import numpy as np
import scipy as sc

pathname1 = "/Users/paulinenicolas/Documents/M2_Data_Science/ML_From_Theory_To_Practice/Project_ML/challenge_training_input_file_predict_the_aesthetic_score_of_a_portrait_by_combining_photo_analysis_and_facial_attributes_analysis/facial_features_train.csv"
pathname2 = "/Users/paulinenicolas/Documents/M2_Data_Science/ML_From_Theory_To_Practice/Project_ML/challenge_output_data_training_file_predict_the_aesthetic_score_of_a_portrait_by_combining_photo_analysis_and_facial_attributes_analysis.csv"

#Features Data
data = pd.read_csv( pathname1, sep = ',')

#Rate (1 to 24) 
data_output = pd.read_csv( pathname2, sep = ';')


#Transform 1-24 Target into binary Target :

def binary_target(df):
    df['BINARY_TARGET'] = np.where(df['TARGET'] >= df['TARGET'].median(), 1, 0)
    df = df.set_index(['ID'])
    return df

