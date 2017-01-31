#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:24:42 2017

@author: estelleaflalo
"""

import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model
# path to the model weights files.
weights_path = '/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/portrait/final_code/vgg16_weights.h5'

# dimensions of our images.
img_width, img_height = 224, 224 #150

train_data_dir = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_dentrees_dentrainement_predire_le_score_esthetique_dun_portrait/pictures_train2_val"
validation_data_dir = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_dentrees_dentrainement_predire_le_score_esthetique_dun_portrait/pictures_train2_valval"
test_dir=("/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichiers_dentrees_de_test_predire_le_score_esthetique_dun_portrait/pictures_test")

nb_train_samples = 1824#2000
nb_validation_samples = 439#800
nb_test_samples=3000
nb_epoch = 2
nb_classes=25

# build the VGG16 network
model = Sequential()

model.add(ZeroPadding2D((1, 1), input_shape=(3,img_width, img_height), dim_ordering="th"))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1', dim_ordering="th"))
model.add(ZeroPadding2D((1, 1), dim_ordering="th"))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2', dim_ordering="th"))
model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering="th"))

model.add(ZeroPadding2D((1, 1), dim_ordering="th"))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1', dim_ordering="th"))
model.add(ZeroPadding2D((1, 1), dim_ordering="th"))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2', dim_ordering="th"))
model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering="th"))

model.add(ZeroPadding2D((1, 1), dim_ordering="th"))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1', dim_ordering="th"))
model.add(ZeroPadding2D((1, 1), dim_ordering="th"))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2', dim_ordering="th"))
model.add(ZeroPadding2D((1, 1), dim_ordering="th"))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3', dim_ordering="th"))
model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering="th"))

model.add(ZeroPadding2D((1, 1), dim_ordering="th"))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1', dim_ordering="th"))
model.add(ZeroPadding2D((1, 1), dim_ordering="th"))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2', dim_ordering="th"))
model.add(ZeroPadding2D((1, 1), dim_ordering="th"))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3', dim_ordering="th"))
model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering="th"))

model.add(ZeroPadding2D((1, 1), dim_ordering="th"))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1', dim_ordering="th"))
model.add(ZeroPadding2D((1, 1), dim_ordering="th"))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2', dim_ordering="th"))
model.add(ZeroPadding2D((1, 1), dim_ordering="th"))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3', dim_ordering="th"))
model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering="th"))


# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
#assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]

    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')


# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(nb_classes, activation='softmax'))



top_model.save_weights('/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/portrait/final_code/top_model.h5') 

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning

top_model.load_weights('/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/portrait/final_code/top_model.h5')

# add the model on top of the convolutional base
model.add(top_model)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:15]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss="categorical_crossentropy",
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['mean_squared_logarithmic_error', 'accuracy'])

# prepare data augmentation configuration

train_datagen = ImageDataGenerator(shear_range=0.3, zoom_range=0.3, rotation_range=0.3,dim_ordering="th")
validation_datagen = ImageDataGenerator(dim_ordering="th")

print('trainning')
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical')
  

print('testing')
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)


from collections import OrderedDict
class_dictionary = train_generator.class_indices
sorted_class_dictionary = OrderedDict(sorted(class_dictionary.items()))
sorted_class_dictionary = sorted_class_dictionary.values()
print(sorted_class_dictionary)

# Fine-tuning the model:
model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)
print("Model fitted")

model.save("/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/my_model_2ep.h5")
model.save_weights("/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/my_model_weights_2ep.h5")

print("Model and weights saved")

#TEST
test_dir=("/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichiers_dentrees_de_test_predire_le_score_esthetique_dun_portrait/test_DELETE")

def spearman_error(y_true, y_pred):
    y_true_rank = rankdata(y_true)
    y_pred_rank = rankdata(y_pred)
    square_distance = np.dot((y_pred_rank - y_true_rank).T, (y_pred_rank - y_true_rank))
    accuracy = 1 - 6*square_distance/(y_pred_rank.shape[0]*(y_pred_rank.shape[0]**2 - 1))
    return accuracy
    
test_datagen = ImageDataGenerator(dim_ordering="th")
test_generator = test_datagen.flow_from_directory(test_dir, batch_size = 32, target_size =(img_width, img_height), class_mode='categorical')
print (test_generator.filenames)
test_data_features = model.predict_generator(test_generator,12)#nb_test_samples )
np.save(open('test_data_features.npy','wb'), test_data_features)
test_data = np.load(open('test_data_features.npy', 'rb'))
#test_data