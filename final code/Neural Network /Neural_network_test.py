#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 15:48:06 2017

@author: paulinenicolas
"""
# We could not upload the folders with the images on Git because it is to heavy. We let here the ends of the paths
pathname = "../challenge_training_input_file_predict_the_aesthetic_score_of_a_portrait_by_combining_photo_analysis_and_facial_attributes_analysis/pictures_train/*.jpg"
pathresult =  "../challenge_output_data_training_file_predict_the_aesthetic_score_of_a_portrait_by_combining_photo_analysis_and_facial_attributes_analysis.csv"
path_test = "/../challenge_testing_input_file_predict_the_aesthetic_score_of_a_portrait_by_combining_photo_analysis_and_facial_attributes_analysis/pictures_test/*.jpg"

from keras.models import Model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation
from PIL import Image
from keras.optimizers import SGD
import cv2, numpy as np
import glob
import pandas as pd
nb_classes = 25

def vgg_face(weights_path=None):
    img = Input(shape=(3, 224, 224))

    pad1_1 = ZeroPadding2D(padding=(1, 1))(img)
    conv1_1 = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(pad1_1)
    pad1_2 = ZeroPadding2D(padding=(1, 1))(conv1_1)
    conv1_2 = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(pad1_2)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2)

    pad2_1 = ZeroPadding2D((1, 1))(pool1)
    conv2_1 = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(pad2_1)
    pad2_2 = ZeroPadding2D((1, 1))(conv2_1)
    conv2_2 = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(pad2_2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2_2)

    pad3_1 = ZeroPadding2D((1, 1))(pool2)
    conv3_1 = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(pad3_1)
    pad3_2 = ZeroPadding2D((1, 1))(conv3_1)
    conv3_2 = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(pad3_2)
    pad3_3 = ZeroPadding2D((1, 1))(conv3_2)
    conv3_3 = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(pad3_3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3_3)

    pad4_1 = ZeroPadding2D((1, 1))(pool3)
    conv4_1 = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(pad4_1)
    pad4_2 = ZeroPadding2D((1, 1))(conv4_1)
    conv4_2 = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(pad4_2)
    pad4_3 = ZeroPadding2D((1, 1))(conv4_2)
    conv4_3 = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(pad4_3)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4_3)

    pad5_1 = ZeroPadding2D((1, 1))(pool4)
    conv5_1 = Convolution2D(512, 3, 3, activation='relu', name='conv5_1')(pad5_1)
    pad5_2 = ZeroPadding2D((1, 1))(conv5_1)
    conv5_2 = Convolution2D(512, 3, 3, activation='relu', name='conv5_2')(pad5_2)
    pad5_3 = ZeroPadding2D((1, 1))(conv5_2)
    conv5_3 = Convolution2D(512, 3, 3, activation='relu', name='conv5_3')(pad5_3)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5_3)


    fc6 = Convolution2D(4096, 7, 7, activation='relu', name='fc6')(pool5)
    fc6_drop = Dropout(0.5)(fc6)
    fc7 = Convolution2D(4096, 1, 1, activation='relu', name='fc7')(fc6_drop)
    fc7_drop = Dropout(0.5)(fc7)
    fc8 = Convolution2D(nb_classes, 1, 1, name='fc8')(fc7_drop)
    flat = Flatten()(fc8)
    out = Activation('softmax')(flat)

    model = Model(input=img, output=out)

    if weights_path:
        model.load_weights(weights_path,by_name=True)
    
    return model


model = vgg_face('/Users/paulinenicolas/downloads/mymodel_4000pic_3epoch.h5')


###Building matrix of features###
import re
numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
    

i=0
for img in sorted(glob.glob(path_test), key=numericalSort):
    im = cv2.resize(cv2.imread(img), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    if i==0:
        im_tot_test = im
    else:
        im_tot_test = np.concatenate((im_tot_test, im), axis=0)
    print (int(os.path.splitext(os.path.basename(img))[0]))
    i+=1


#Predictin the testing set    
y_pred = model.predict(im_tot_test)
y_pred2 = np.argmax(y_pred,axis=1)
id = np.arange(10001, 13001)
id.reshape(-1,1)

y_pred3 = np.vstack((id, y_pred2))
y_pred3 = y_pred3.T

np.save('/Users/paulinenicolas/Downloads/array_test_deep', im_tot_test)

df = pd.DataFrame(y_pred3, columns=['ID','TARGET'])
df['ID'] = df.index

#os.getcwd() is portrait

df.to_csv(os.getcwd()+str('/dataframes/result_deep.csv'), sep=';', index = False)
