#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:56:14 2017
@author: paulinenicolas
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing


pathname1 = "/Users/paulinenicolas/Documents/M2_Data_Science/ML_From_Theory_To_Practice/Project_ML/portrait/result.csv"
pathname2 = "/Users/paulinenicolas/Documents/M2_Data_Science/ML_From_Theory_To_Practice/Project_ML/challenge_output_data_training_file_predict_the_aesthetic_score_of_a_portrait_by_combining_photo_analysis_and_facial_attributes_analysis.csv"

#Features Data
X_df = pd.read_csv( pathname1, sep = ',')


#Rate (1 to 24) 
y_df = pd.read_csv( pathname2, sep = ';')

#Feature extractor function (only selecting the impact function)
labels = np.array(['0', '1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24'])


class FeatureExtractorClf(object):
    def __init__(self):

        pass

    def fit(self, X_df, y_df):
        pass
    
    def transform(self, X_df, y_df):
        
        #selecting only features that are interesting for the prdiction (i.e impact feature)
        XX = X_df.copy()
        XX = XX.set_index(['ID'])
        XX = XX.drop(['Unnamed: 0', 'd_p1', 'd_p2', 'd_p3', 'd_p4', 'd_l1', 'd_l2', 'd_l3', 'd_l4'], axis=1)
        #XX = XX.select_dtypes(include=['float64', 'int'])
        XX = np.array(XX)
        # mean = 0 ; standard deviation = 1.0
        scaler = preprocessing.StandardScaler()
        XX = scaler.fit_transform(XX)
        
        #Transforming the actual target into a binary target
        yy = y_df.copy()
        #yy['BINARY_TARGET'] = np.where(yy['TARGET'] >= yy['TARGET'].median(), 1, 0)
        yy = yy.set_index(['ID'])


        #del yy['TARGET']
        yy = np.array(yy)

        return XX, yy


    
#Fitting model function (SVM)
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


class Classifier(BaseEstimator):
    def __init__(self, C):
        self.n_components = 10
        self.C = C
        self.clf = Pipeline([('pca', PCA(n_components=self.n_components)), 
                             ('clf', SVC(C = self.C, probability = True))]) 

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)    
        
#Train Test model 

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


skf = ShuffleSplit(n_splits=2, test_size=0.2, random_state=57)  
skf_is = list(skf.split(X_df))[0]

def train_test_model_clf(X_df, y_df, skf_is, FeatureExtractor, Classifier):
    train_is, test_is = skf_is
    X_train_df = X_df.iloc[train_is].copy()                                  
    y_train_df = y_df.iloc[train_is].copy()
    X_test_df = X_df.iloc[test_is].copy()                                    
    y_test_df = y_df.iloc[test_is].copy() 

    # Feature extraction
    fe_clf = FeatureExtractor
    fe_clf.fit(X_train_df, y_train_df)
    X_train_array_clf, y_train_array_clf = fe_clf.transform(X_train_df, y_train_df)
    X_test_array_clf, y_test_array_clf = fe_clf.transform(X_test_df, y_test_df)

   
    # Train
    clf = Classifier
    clf.fit(X_train_array_clf, y_train_array_clf)
    
    # Test          
    y_proba_clf = clf.predict_proba(X_test_array_clf)                        
    y_pred_clf = labels[np.argmax(y_proba_clf, axis=1)]                      
    accuracy = spearman_error(y_test_array_clf, y_pred_clf)      
    print(y_pred_clf[:10], y_test_array_clf[:10])                                   
    return accuracy

#Definition of the error :
from scipy.stats import rankdata
    
def spearman_error(y_true, y_pred):
    y_true_rank = rankdata(y_true)
    y_pred_rank = rankdata(y_pred)
    square_distance = np.dot((y_pred_rank - y_true_rank).T, (y_pred_rank - y_true_rank))
    accuracy = 1 - 6*square_distance/(y_pred_rank.shape[0]*(y_pred_rank.shape[0]**2 - 1))

    return accuracy

    
#Cross Validation in order to find  the value of C which predict the best   
C = [0.001, 0.01, 0.1, 1]
 
 
accuracies = []
 
for i in range(len(C)):
     
    reg = Classifier(C[i])
    FeatureExtractor = FeatureExtractorClf()
    skf = ShuffleSplit(n_splits=2, test_size=0.2, random_state=57)  
    skf_is = list(skf.split(X_df))[0]
 
    accuracies.append(train_test_model_clf(X_df, y_df, skf_is, FeatureExtractor, reg))
     
print(accuracies)

