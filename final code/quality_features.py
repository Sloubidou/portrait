#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:37:06 2017

@author: paulinenicolas
"""
import pandas as pd
import cv2
import numpy as np
import pandas as pd

<<<<<<< HEAD
pathname1 = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_dentrees_dentrainement_predire_le_score_esthetique_dun_portrait/facial_features_train.csv"
pathname2 = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_de_sortie_dentrainement_predire_le_score_esthetique_dun_portrait.csv"
=======
#pathname1 = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_dentrees_dentrainement_predire_le_score_esthetique_dun_portrait/facial_features_train.csv"
#pathname2 = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_de_sortie_dentrainement_predire_le_score_esthetique_dun_portrait.csv"
>>>>>>> 52a051e86e5f0a4e43cb82ded0f502343b06c8ed
#pathname1 = "/Users/paulinenicolas/Documents/M2_Data_Science/ML_From_Theory_To_Practice/Project_ML/challenge_training_input_file_predict_the_aesthetic_score_of_a_portrait_by_combining_photo_analysis_and_facial_attributes_analysis/facial_features_train.csv"
#pathname2 = "/Users/paulinenicolas/Documents/M2_Data_Science/ML_From_Theory_To_Practice/Project_ML/challenge_output_data_training_file_predict_the_aesthetic_score_of_a_portrait_by_combining_photo_analysis_and_facial_attributes_analysis.csv"

#pathname = "/Users/paulinenicolas/Documents/M2_Data_Science/ML_From_Theory_To_Practice/Project_ML/challenge_training_input_file_predict_the_aesthetic_score_of_a_portrait_by_combining_photo_analysis_and_facial_attributes_analysis/pictures_train/*.jpg"

#df = pd.read_csv(pathname1, sep = ',')
#df_score = pd.read_csv(pathname2, sep = ';')


#image_br= "/Users/paulinenicolas/Documents/M2_Data_Science/ML_From_Theory_To_Practice/Project_ML/challenge_training_input_file_predict_the_aesthetic_score_of_a_portrait_by_combining_photo_analysis_and_facial_attributes_analysis/pictures_train/1031.jpg"
#image_nbr= "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/267.jpg"


def blurry_tot(image) :
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


def blurry_face(im, ID_im):
    i=df.loc[df['ID']==ID_im].index[0]
    x_0 =int(df['x0'].ix[i]*im.shape[1])
    y_0 =int(df['y0'].ix[i]*im.shape[0])
    x_1 = int((df['x0'].ix[i]+df['width'].ix[i])*im.shape[1])
    y_1 = int((df['y0'].ix[i]+df['height'].ix[i])*im.shape[0])

    face_im = im[y_0:y_1, x_0:x_1] #image cropped around the face
    
    gray = cv2.cvtColor(face_im, cv2.COLOR_BGR2GRAY)
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm

def blurry_background(im,ID_im):
    i=df.loc[df['ID']==ID_im].index[0]
    x_0 =int(df['x0'].ix[i]*im.shape[1])
    y_0 =int(df['y0'].ix[i]*im.shape[0])
    x_1 = int((df['x0'].ix[i]+df['width'].ix[i])*im.shape[1])
    y_1 = int((df['y0'].ix[i]+df['height'].ix[i])*im.shape[0])
    backg_im1 = im[y_0:y_1 , :1+x_0]
    backg_im2 = im[y_0:y_1 , x_1-1:]
    backg_im3 = im[y_1-1: , :]
    backg_im4 = im[:y_0 +1, :]#image cropped around the face
    
    gray = np.concatenate((cv2.cvtColor(backg_im1 , cv2.COLOR_BGR2GRAY).ravel(),cv2.cvtColor(backg_im2 , cv2.COLOR_BGR2GRAY).ravel(),cv2.cvtColor(backg_im3 , cv2.COLOR_BGR2GRAY).ravel(),cv2.cvtColor(backg_im4 , cv2.COLOR_BGR2GRAY).ravel()),axis=0)
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm#fm

########### Sharpness function ###########
