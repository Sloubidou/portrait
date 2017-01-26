#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:54:37 2017

@author: paulinenicolas
"""

import cv2
import glob
import os
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error


from quality_features import blurry
from spatial_features import dist_rule_thirds, get_face_center,face_ratio, eye_position


#popo
pathname = "/Users/paulinenicolas/Documents/M2_Data_Science/ML_From_Theory_To_Practice/Project_ML/challenge_training_input_file_predict_the_aesthetic_score_of_a_portrait_by_combining_photo_analysis_and_facial_attributes_analysis/pictures_train/*.jpg"
#dom
pathname = '/Users/domitillecoulomb/Documents/DATA_SCIENCE/MachineLearning/Projet/pictures_test/*.jpg'
path_data = '/Users/domitillecoulomb/Documents/DATA_SCIENCE/MachineLearning/Projet/facial_features_train.csv'

#slou
pathname = '/home/slou/Documents/M2/semestre1/ML_project/pictures_train/*.jpg'
pathresult = '/home/slou/Documents/M2/semestre1/ML_project/output_train.csv'
path_data = '/home/slou/Documents/M2/semestre1/ML_project/facial_features_train.csv'

data = pd.read_csv(path_data)
result = pd.read_csv(pathresult, sep =";")

#adding our features to a global dataframe with the picture id
df = pd.DataFrame(columns=('ID', 'blur'
                            ,'d_p1','d_p2','d_p3','d_p4'
                            ,'d_l1','d_l2','d_l3', 'd_l4'
                            , 'face_ratio'
                            , 'left_eye_lvl', 'right_eye_level'))

i = 0
for img in glob.glob(pathname):
    
    #Read the image
    image = cv2.imread(img)
    idx = int(os.path.splitext(os.path.basename(img))[0])
    
    #get saptial features
    a,b = get_face_center(data['x0'].ix[idx-1], data['width'].ix[idx-1], data['y0'].ix[idx-1], data['height'].ix[idx-1])
                        
    drt = dist_rule_thirds(a,b)
    
    eyes_level = eye_position(data['left_eye_y'].ix[idx-1], data['right_eye_y'].ix[idx-1])
    
    brightness = brightness(image)
    
    
    #Filling the df line by line
    df.loc[i] = [os.path.splitext(os.path.basename(img))[0]
                , blurry(image)
                , drt[0], drt[1],drt[2],drt[3],drt[4],drt[5],drt[6],drt[7]
                , face_ratio(data['width'].ix[idx-1], data['height'].ix[idx-1])
                , eyes_level[0], eyes_level[1]
                , ]
    i+=1

#Split data
#X_train, X_test, y_train, y_test = train_test_split(x_selection_position, y_position_impact, train_size=0.8, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(df, result['TARGET'], train_size=0.8, random_state=0)
print("Nb d'échantillons d'apprentissage :  {}".format(X_train.shape[0]))
print("Nb d'échantillons de validation :    {}".format(X_test.shape[0]))

#Train the model 
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(X_train, y_train)

#Prediction
y_pred = svr_rbf.predict(X_test)

# accuracy : mean square error
print("Mean Squared error: {}", mean_squared_error(y_test, y_pred))
print(y_pred[:10])
print(np.std(y_pred))