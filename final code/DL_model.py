#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 21:46:40 2017

@author: estelleaflalo
"""

import cv2
import math
import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16
from imagenet_utils import decode_predictions, preprocess_input
pathname="/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_dentrees_dentrainement_predire_le_score_esthetique_dun_portrait/pictures_train/*.jpg"
pathresult = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_de_sortie_dentrainement_predire_le_score_esthetique_dun_portrait.csv"
path_data = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_dentrees_dentrainement_predire_le_score_esthetique_dun_portrait/facial_features_train.csv"


df = pd.read_csv(path_data, sep = ',')
df_score = pd.read_csv(pathresult, sep = ';')

im_test="/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_dentrees_dentrainement_predire_le_score_esthetique_dun_portrait/pictures_train/25.jpg"

def preprocess_input(x, dim_ordering='default'):
    """Preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input Numpy tensor, 4D.
        dim_ordering: data format of the image tensor.
    # Returns
        Preprocessed tensor.
    """
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
        # Zero-center by mean pixel
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
    return x
width=[]
height=[]
for img in glob.glob(pathname):
    width.append(cv2.imread(img).shape[1])
    height.append(cv2.imread(img).shape[0])

max_width=max(width) #6381
max_height=max(height) #6016

for img in glob.glob(pathname):
    
    #im=cv2.imread(img)
    #im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #resize=np.zeros((max_height,max_width))
    #resize[:im.shape[0],:im.shape[1]]=im
    #print resize, resize.shape

model = VGG16(weights="imagenet")

im=cv2.imread(im_test)[:224,:224,:]
x = np.expand_dims(im, axis=0)
preds = model.predict(x)


