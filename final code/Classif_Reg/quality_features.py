#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pandas as pd


def blurry_tot(image) :
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


def blurry_face(im,x0, y0, width, height ):
    x_0 =int(x0*im.shape[1])
    y_0 =int(y0*im.shape[0])
    x_1 = int((x0+width)*im.shape[1])
    y_1 = int((y0+height)*im.shape[0])

    face_im = im[y_0:y_1, x_0:x_1] #image cropped around the face

    gray = cv2.cvtColor(face_im, cv2.COLOR_BGR2GRAY)
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm

def blurry_background(im,x0, y0, width, height):
    x_0 = int(x0*im.shape[1])
    y_0 = int(y0*im.shape[0])
    x_1 = int((x0+width)*im.shape[1])
    y_1 = int((y0+height)*im.shape[0])
    
    backg_im = []
    gray = []
    fm = []
    
    backg_im.append(im[y_0:y_1 , :x_0+1])
    backg_im.append(im[y_0:y_1 , x_1-1:])
    backg_im.append(im[y_1-1: , :])
    backg_im.append(im[:1+y_0 , :])#image cropped around the face
    
    gray = [cv2.cvtColor(backg_el , cv2.COLOR_BGR2GRAY) for backg_el in backg_im]
    fm = [cv2.Laplacian(gray_el, cv2.CV_64F).var() for gray_el in gray]        
                 
    # compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
    
    return np.mean(fm)

########### Sharpness function ###########
if __name__ == '__main__':
    pathname1 = "/Users/paulinenicolas/Documents/M2_Data_Science/ML_From_Theory_To_Practice/Project_ML/challenge_training_input_file_predict_the_aesthetic_score_of_a_portrait_by_combining_photo_analysis_and_facial_attributes_analysis/facial_features_train.csv"
    pathname_pic = "/Users/paulinenicolas/Documents/M2_Data_Science/ML_From_Theory_To_Practice/Project_ML/challenge_training_input_file_predict_the_aesthetic_score_of_a_portrait_by_combining_photo_analysis_and_facial_attributes_analysis/pictures_train/1167.jpg"
    df =  pd.read_csv(pathname1, sep = ',')
    df_1167 = df.loc[df['ID']== 1167]
    df_1167['x0']
    im = cv2.imread(pathname_pic)
    l= blurry_background(im, df_1167['x0'], df_1167['y0'], df_1167['width'], df_1167['height'])

