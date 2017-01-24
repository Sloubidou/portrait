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


pathname = "/Users/paulinenicolas/Documents/M2_Data_Science/ML_From_Theory_To_Practice/Project_ML/challenge_training_input_file_predict_the_aesthetic_score_of_a_portrait_by_combining_photo_analysis_and_facial_attributes_analysis/pictures_train/*.jpg"


#adding our features to a global dataframe with the picture id
df = pd.DataFrame(columns=('ID', 'blur'))

i = 0
for img in glob.glob(pathname):
    
    #Read the image
    image = cv2.imread(img)
    
    #Filling the df line by line
    df.loc[i] = [os.path.splitext(os.path.basename(img))[0], blurry(image)]