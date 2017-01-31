#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:56:14 2017
@author: paulinenicolas
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import pdb

#pathname for the feature extractor dataframe of the trianong set
pathname1 = "./dataframes/dataframe.csv"

#pathname of the target of the training set
pathname2 = "./input_data_available/challenge_output_data_training_file_predict_the_aesthetic_score_of_a_portrait_by_combining_photo_analysis_and_facial_attributes_analysis.csv"
#Features Data
X_df = pd.read_csv( pathname1, sep = ',')
X_df = X_df.sort('ID')
X_df

#Rate (1 to 24) 
y_df = pd.read_csv( pathname2, sep = ';')

#Feature extractor function (only selecting the impact function)
class FeatureExtractorReg(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_df):
        pass
    
    def transform(self, X_df, y_df):
        
        #selecting only features that are interesting for the prdiction (i.e impact feature)
        XX = X_df.copy()
        XX = XX.set_index(['ID'])
        XX = XX.drop(['Unnamed: 0'], axis=1)
        #XX = XX.select_dtypes(include=['float64', 'int'])
        XX = np.array(XX)
        # mean = 0 ; standard deviation = 1.0
        XX = preprocessing.scale(XX)
        
        #Transforming the actual target into a binary target
        yy = y_df.copy()
        #yy['BINARY_TARGET'] = np.where(yy['TARGET'] >= yy['TARGET'].median(), 1, 0)
        yy = yy.set_index(['ID'])


        #del yy['TARGET']
        yy = yy.values.tolist()
        yy = [item for sublist in yy for item in sublist]

        return XX, yy


    
#Fitting model function (SVM)
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVR
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


class Regressor(BaseEstimator):
    def __init__(self, C):
        self.n_components = 10
        self.C = C
        self.reg = SVR( C = self.C)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)

    def predict_proba(self, X):
        return self.reg.predict_proba(X)    
        
#Train Test model 

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


skf = ShuffleSplit(n_splits=2, test_size=0.2, random_state=57)  
skf_is = list(skf.split(X_df))[0]

def train_test_model_reg(X_df, y_df, skf_is, FeatureExtractor, Regressor):
    train_is, test_is = skf_is
    X_train_df = X_df.iloc[train_is].copy()                                  
    y_train_df = y_df.iloc[train_is].copy()
    X_test_df = X_df.iloc[test_is].copy()                                    
    y_test_df = y_df.iloc[test_is].copy() 

    # Feature extraction
    fe_reg = FeatureExtractor
    fe_reg.fit(X_train_df, y_train_df)
    X_train_array_reg, y_train_array_reg = fe_reg.transform(X_train_df, y_train_df)
    X_test_array_reg, y_test_array_reg = fe_reg.transform(X_test_df, y_test_df)

   
    # Train
    reg = Regressor
    reg.fit(X_train_array_reg, y_train_array_reg)
    
    # Test          
    #y_proba_clf = clf.predict_proba(X_test_array_clf)                        
    #y_pred_clf = labels[np.argmax(y_proba_clf, axis=1)] 
    y_pred_reg = reg.predict(X_test_array_reg).astype(int)                
    accuracy = spearman_correlation(y_test_array_reg, y_pred_reg)      
    #print(y_pred_reg[:10], y_test_array_reg[:10])                                   
    return accuracy

#Definition of the error :
from scipy.stats import rankdata
    
def spearman_correlation(y_true, y_pred):
    y_true_rank = rankdata(y_true)
    y_pred_rank = rankdata(y_pred)
    square_distance = np.dot((y_pred_rank - y_true_rank).T, (y_pred_rank - y_true_rank))
    accuracy = 1 - 6*square_distance/(y_pred_rank.shape[0]*(y_pred_rank.shape[0]**2 - 1))

    return accuracy

    
#Cross Validation in order to find  the value of C which predict the best   
C = [1, 10, 100]
 
 
accuracies = []
print("Cross Validation on SVR to chose the hyperparameter")
 
for i in range(len(C)):
     
    reg = Regressor(C[i])
    FeatureExtractor = FeatureExtractorReg()
    skf = ShuffleSplit(n_splits=2, test_size=0.2, random_state=57)  
    skf_is = list(skf.split(X_df))[0]
    acc = train_test_model_reg(X_df, y_df, skf_is, FeatureExtractor, reg)
    print('for C = ', C[i], ' spearman correlation = ', acc)
    accuracies.append(acc)



###True prediction a la fin du fichier de cross val
print("Calculating target for testing set for SVR with C=10..")
reg = Regressor(10)
FeatureExtractor = FeatureExtractorReg()
skf = ShuffleSplit(n_splits=2, test_size=0.2, random_state=57)  
skf_is = list(skf.split(X_df))[0]
acc = train_test_model_reg(X_df, y_df, skf_is, FeatureExtractor, reg)
#path until feature matrix of testing set
pathname_result = "./dataframes/dataframe_test.csv"
X_test_df = pd.read_csv(pathname_result, sep = ',')
X_test = X_test_df.copy()

#Feature extraction (perhaps need to be changed)
X_test = X_test.set_index(['ID'])
X_test = X_test.drop(['Unnamed: 0'], axis=1)
X_test = np.array(X_test)
X_test = preprocessing.scale(X_test)

y_pred = reg.predict(X_test).astype(int)
id = np.arange(10001, 13001)
id.reshape(-1,1)

y_pred2 = np.vstack((id, y_pred))
y_pred2 = y_pred2.T


df_pred = pd.DataFrame(y_pred2, columns=['ID','TARGET'])

#path where to save your csv file
df_pred.to_csv('./dataframes/target_test_svm.csv', sep=';', index = False)
print("Done")

