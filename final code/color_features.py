 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:37:06 2017

@author: paulinenicolas
"""

import cv2
import math
import numpy as np
import pandas as pd
#pathname1 = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_dentrees_dentrainement_predire_le_score_esthetique_dun_portrait/facial_features_train.csv"
#pathname2 = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_de_sortie_dentrainement_predire_le_score_esthetique_dun_portrait.csv"


#df = pd.read_csv(pathname1, sep = ',')
#df_score = pd.read_csv(pathname2, sep = ';')


#image_br= "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/109.jpg"
#image_nbr= "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/267.jpg"
#image_24= "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/6305.jpg"

def brightness( im ):
    b,g,r = np.mean(im, axis=(0,1))
    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
    
def hsv_im(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h_m,s_m,v_m = np.mean(hsv, axis=(0,1))
    h_std,s_std,v_std = np.std(hsv, axis=(0,1)) 
    return (h_m,s_m,v_m ,h_std,s_std,v_std)
    
def hsv_face(im,x0, y0, width, height ):
    x_0 =int(x0*im.shape[1])
    y_0 =int(y0*im.shape[0])
    x_1 = int((x0+width)*im.shape[1])
    y_1 = int((y0+height)*im.shape[0])

    face_im = im[y_0:y_1, x_0:x_1] #image cropped around the face
    hsv_face = cv2.cvtColor(face_im, cv2.COLOR_BGR2HSV)
    h_m,s_m,v_m = np.mean(hsv_face, axis=(0,1))
    h_std,s_std,v_std = np.std(hsv_face, axis=(0,1))
    return (h_m,s_m,v_m ,h_std,s_std,v_std)
    
<<<<<<< HEAD
def hsv_background(im,x0, y0, width, height ):
    x_0 =int(x0*im.shape[1])
    y_0 =int(y0*im.shape[0])
    x_1 = int((x0+width)*im.shape[1])
    y_1 = int((y0+height)*im.shape[0])
    backg_im1 = im[y_0:y_1 , :x_0+1]
    backg_im2 = im[y_0:y_1 , x_1-1:]
    backg_im3 = im[y_1-1: , :]
    backg_im4 = im[:1+y_0 , :]#image cropped around the face
    h_backg= np.concatenate((cv2.cvtColor(backg_im1 , cv2.COLOR_BGR2HSV)[:,:,0].ravel(),cv2.cvtColor(backg_im2 , cv2.COLOR_BGR2HSV)[:,:,0].ravel(),cv2.cvtColor(backg_im3 , cv2.COLOR_BGR2HSV)[:,:,0].ravel(),cv2.cvtColor(backg_im4 , cv2.COLOR_BGR2HSV)[:,:,0].ravel()),axis=0)
    s_backg= np.concatenate((cv2.cvtColor(backg_im1 , cv2.COLOR_BGR2HSV)[:,:,1].ravel(),cv2.cvtColor(backg_im2 , cv2.COLOR_BGR2HSV)[:,:,1].ravel(),cv2.cvtColor(backg_im3 , cv2.COLOR_BGR2HSV)[:,:,1].ravel(),cv2.cvtColor(backg_im4 , cv2.COLOR_BGR2HSV)[:,:,1].ravel()),axis=0)
    v_backg= np.concatenate((cv2.cvtColor(backg_im1 , cv2.COLOR_BGR2HSV)[:,:,2].ravel(),cv2.cvtColor(backg_im2 , cv2.COLOR_BGR2HSV)[:,:,2].ravel(),cv2.cvtColor(backg_im3 , cv2.COLOR_BGR2HSV)[:,:,2].ravel(),cv2.cvtColor(backg_im4 , cv2.COLOR_BGR2HSV)[:,:,2].ravel()),axis=0)    
    h_m,s_m,v_m = np.mean(h_backg),np.mean(s_backg),np.mean(v_backg)
    h_std,s_std,v_std = np.std(h_backg),np.std(s_backg),np.std(v_backg)
    return (h_m,s_m,v_m ,h_std,s_std,v_std)
    
#def V_back_fore_diff(im,ID_im):
#    return abs(hsv_face(im,ID_im )[2] - hsv_background(im,ID_im)[2])
    

#im=cv2.imread(image_br)
#im2=cv2.imread(image_nbr) 
#print (brightness(im),brightness(im2))

#ID_im = 267
#im=cv2.imread(image_nbr)
im=cv2.imread(image_br)
im2=cv2.imread(image_nbr) 
print (brightness(im),brightness(im2))

ID_im = 267
im=cv2.imread(image_nbr)
print (hsv_background(im,ID_im),hsv_face(im,ID_im))


image_br= "/Users/paulinenicolas/Documents/M2_Data_Science/ML_From_Theory_To_Practice/Project_ML/challenge_training_input_file_predict_the_aesthetic_score_of_a_portrait_by_combining_photo_analysis_and_facial_attributes_analysis/pictures_train/1031.jpg"
ID_im = 1031
im=cv2.imread(image_br)
print (hsv_background(im,ID_im),hsv_face(im,ID_im))

print (V_back_fore_diff(im,ID_im))
#print hsv_background(im,ID_im),hsv_face(im,ID_im)

#ID_im = 109
#im=cv2.imread(image_br)
#print hsv_background(im,ID_im),hsv_face(im,ID_im)

#ID_im = 6305
#im=cv2.imread(image_24)
#print (hsv_background(im,ID_im),hsv_face(im,ID_im))
#print V_back_fore_diff(im,ID_im)
#### Contrast function #####
#### Clor Function
