#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 12:26:19 2017

@author: paulinenicolas
"""

## Befor Running, need to upload vgg-face-keras.h5 at https://gist.github.com/EncodeTS/6bbe8cb8bebad7a672f0d872561782d9
# We could not upload the folders with the images on Git because it is to heavy. We let here the ends of the paths
pathname = "./pictures_train/*.jpg"
pathresult =  "./dataframs/output_train.csv"
path_test = "./pictures_test/*.jpg"

from keras.models import Model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation
from PIL import Image
from keras.optimizers import SGD
import cv2, numpy as np
import glob
import pandas as pd
import os
nb_classes = 24

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
    fc8 = Convolution2D(2622, 1, 1, name='fc8')(fc7_drop)
    flat = Flatten()(fc8)
    out = Activation('softmax')(flat)

    model = Model(input=img, output=out)

    if weights_path:
        #fitting with weights from a pre-trained model
        model.load_weights(weights_path,by_name=True)
        
        #removing the three last layers
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        #adding new layers including a convlutionnal with 24 neurons for the output (nb of classes)
        fc8 = Convolution2D(nb_classes, 1, 1, name='fc8')(fc7_drop)
        flat = Flatten()(fc8)
        out = Activation('softmax')(flat)
        model = Model(input=img, output=out)

    
    return model

#Fit the mode with a pre-trianed model https://gist.github.com/EncodeTS/6bbe8cb8bebad7a672f0d872561782d9
model = vgg_face('./vgg-face-keras.h5')


#Function used to get the image sorted numerically
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
    

#Training the last layer on 500 images of our datset
#Transforming pictures into matrices of features
i=0
for img in sorted(glob.glob(pathname), key=numericalSort):
    if i<=400:
        print(i)
        i+=1
    else:
        
        im = cv2.resize(cv2.imread(img), (224, 224)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        if i==401:
            im_tot = im
        else:
            im_tot = np.concatenate((im_tot, im), axis=0)
        print (int(os.path.splitext(os.path.basename(img))[0]))
        i+=1
        if i>1400:
            break

#loading the true labels
result = pd.read_csv(pathresult, sep =";")
result = result.set_index(['ID'])
result = result.values.tolist()
result = [item for sublist in result for item in sublist]
result = result[401:1401]

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.metrics import mean_squared_error
import keras.backend as K


#transform labels into categorical vectors
result_cat = to_categorical(result, nb_classes)
result_cat.reshape((-1,1))

#Splitting the dataset in a train/test dataset
X_train, X_test, y_train, y_test = train_test_split(
    im_tot, result, test_size=0.15, random_state=37)


#Except for the three last layeres, we do not the others to be trained, 
#So we set .trainable = False      
for i in range(len(model.layers)-3):
    model.layers[i].trainable = False
    

#Defining the optimization method we want
model.compile(optimizer='adagrad',  
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Fitting to the training set
print('Fitting the model...')
model.fit(X_train, y_train, nb_epoch=2, batch_size=32, validation_data=(X_test, y_test))

print('Saving weights...')
#Saving the weights after training phase
model.save_weights('./final\ code/Neural\ Network/mymodel_500pic_2epoch.h5')

#### Prediction ####

#We predict on 100 images
i=0
for img in sorted(glob.glob(pathname), key=numericalSort):
    if i<=2000:
        print(i)
        i+=1
    else:
        
        im = cv2.resize(cv2.imread(img), (224, 224)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        if i==2001:
            im_result = im
        else:
            im_result = np.concatenate((im_result, im), axis=0)
        print (i)
        i+=1
        if i>2100:
            break
 

#Taking argmax of category vector for y_pred       
y_pred = model.predict(im_result)
y_pred2 = np.argmax(y_pred,axis=1)

y_true = pd.read_csv(pathresult, sep =";")
y_true = y_true.set_index(['ID'])
y_true = y_true.values.tolist()
y_true = [item for sublist in y_true for item in sublist]
y_true = y_true[2001:2101]


#definition of the spearman error, that we want to minimize
from scipy.stats import rankdata
def spearman_error(y_true, y_pred):
    y_true_rank = rankdata(y_true)
    y_pred_rank = rankdata(y_pred)
    square_distance = np.dot((y_pred_rank - y_true_rank).T, (y_pred_rank - y_true_rank))
    accuracy = 1 - 6*square_distance/(y_pred_rank.shape[0]*(y_pred_rank.shape[0]**2 - 1))

    return accuracy

#Checking how the code went
print('spearman coorelation =',spearman_error(y_true, y_pred2))
print('y true :', y_true[60:70])
print('y pred :',y_pred2[60:70])


