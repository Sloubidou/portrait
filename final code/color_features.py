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
pathname1 = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_dentrees_dentrainement_predire_le_score_esthetique_dun_portrait/facial_features_train.csv"
pathname2 = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_de_sortie_dentrainement_predire_le_score_esthetique_dun_portrait.csv"


df = pd.read_csv(pathname1, sep = ',')
df_score = pd.read_csv(pathname2, sep = ';')


image_br= "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/109.jpg"
image_nbr= "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/267.jpg"

def brightness( im ):
    b,g,r = np.mean(im, axis=(0,1))
    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
    
def hsv_im(im,ID_im ):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h,s,v = np.mean(hsv, axis=(0,1))    
    
def hsv_face(im,ID_im ):
    i=df.loc[df['ID']==ID_im].index[0]
    x_0 =int(df['x0'].ix[i]*im.shape[1])
    y_0 =int(df['y0'].ix[i]*im.shape[0])
    x_1 = int((df['x0'].ix[i]+df['width'].ix[i])*im.shape[1])
    y_1 = int((df['y0'].ix[i]+df['height'].ix[i])*im.shape[0])

    face_im = im[y_0:y_1, x_0:x_1] #image cropped around the face
    hsv_face = cv2.cvtColor(face_im, cv2.COLOR_BGR2HSV)
    h,s,v = np.mean(hsv_face, axis=(0,1))
    return h, s, v
    
def hsv_background(im,ID_im):
    i=df.loc[df['ID']==ID_im].index[0]
    x_0 =int(df['x0'].ix[i]*im.shape[1])
    y_0 =int(df['y0'].ix[i]*im.shape[0])
    x_1 = int((df['x0'].ix[i]+df['width'].ix[i])*im.shape[1])
    y_1 = int((df['y0'].ix[i]+df['height'].ix[i])*im.shape[0])
    backg_im1 = im[y_0:y_1 , :x_0]
    backg_im2 = im[y_0:y_1 , x_1:]
    backg_im3 = im[y_1: , :]
    backg_im4 = im[:y_0 , :]#image cropped around the face
    hsv_backg1= cv2.cvtColor(backg_im1 , cv2.COLOR_BGR2HSV)
    hsv_backg2= cv2.cvtColor(backg_im2 , cv2.COLOR_BGR2HSV)
    hsv_backg3= cv2.cvtColor(backg_im3 , cv2.COLOR_BGR2HSV)
    hsv_backg4= cv2.cvtColor(backg_im4 , cv2.COLOR_BGR2HSV)
    h,s,v = np.sum(hsv_backg1, axis=(0,1))+np.sum(hsv_backg2, axis=(0,1))+np.sum(hsv_backg3, axis=(0,1))+np.sum(hsv_backg4, axis=(0,1))
    pix_tot = backg_im1.shape[0]*backg_im1.shape[1]+backg_im2.shape[0]*backg_im2.shape[1]+backg_im3.shape[0]*backg_im3.shape[1]+backg_im4.shape[0]*backg_im4.shape[1]
    return h/pix_tot, s/pix_tot, v/pix_tot 
    
def V_back_fore_diff(im,ID_im):
    return abs(hsv_face(im,ID_im )[0] - hsv_background(im,ID_im)[0])
    
im=cv2.imread(image_br)
im2=cv2.imread(image_nbr) 
print brightness(im),brightness(im2)

ID_im = 267
im=cv2.imread(image_nbr)
print hsv_background(im,ID_im),hsv_face(im,ID_im)

ID_im = 109
im=cv2.imread(image_br)
print hsv_background(im,ID_im),hsv_face(im,ID_im)

print V_back_fore_diff(im,ID_im)
#### Contrast function #####
#### Clor Function
