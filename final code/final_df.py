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
from quality_features import blurry
from spatial_features import dist_rule_thirds, get_face_center

#popo
pathname = "/Users/paulinenicolas/Documents/M2_Data_Science/ML_From_Theory_To_Practice/Project_ML/challenge_training_input_file_predict_the_aesthetic_score_of_a_portrait_by_combining_photo_analysis_and_facial_attributes_analysis/pictures_train/*.jpg"
#dom
pathname = '/Users/domitillecoulomb/Documents/DATA_SCIENCE/MachineLearning/Projet/pictures_test/*.jpg'
path_data = '/Users/domitillecoulomb/Documents/DATA_SCIENCE/MachineLearning/Projet/facial_features_train.csv'

data = pd.read_csv(path_data, nrows=20)

#adding our features to a global dataframe with the picture id
df = pd.DataFrame(columns=('ID', 'blur'
                            ,'d_p1','d_p2','d_p3','d_p4'
                            ,'d_l1','d_l2','d_l3', 'd_l4'))

i = 0
for img in glob.glob(pathname):
    
    #Read the image
    image = cv2.imread(img)
    idx = int(os.path.splitext(os.path.basename(img))[0])
    
    #get saptial features
    a,b = get_face_center(image.shape[1], image.shape[0]
                        , data['x0'].ix[idx-1], data['width'].ix[idx-1]
                        , data['y0'].ix[idx-1], data['height'].ix[idx-1])
                        
    drt = dist_rule_thirds(image.shape[1], image.shape[0], a,b)
    
    #Filling the df line by line
    df.loc[i] = [os.path.splitext(os.path.basename(img))[0]
                , blurry(image)
                , drt[0], drt[1],drt[2],drt[3],drt[4],drt[5],drt[6],drt[7]]
    i+=1

