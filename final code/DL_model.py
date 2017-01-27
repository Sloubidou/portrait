import cv2
import math
import numpy as np
import glob
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
pathname="/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_dentrees_dentrainement_predire_le_score_esthetique_dun_portrait/pictures_train/*.jpg"
pathresult = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_de_sortie_dentrainement_predire_le_score_esthetique_dun_portrait.csv"
path_data = "/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_dentrees_dentrainement_predire_le_score_esthetique_dun_portrait/facial_features_train.csv"


data = pd.read_csv(path_data,sep = ',')
result = pd.read_csv(pathresult, sep =";")

im_test="/Users/estelleaflalo/Desktop/M2_Data_Science/First_Period/Machine_Learning_from_Theory_to_Practice/Project/challenge_fichier_dentrees_dentrainement_predire_le_score_esthetique_dun_portrait/pictures_train/25.jpg"


im=cv2.imread(im_test)
im = cv2.resize(im, (224, 224)).astype(np.float32)
im[:,:,0] -= 103.939
im[:,:,1] -= 116.779
im[:,:,2] -= 123.68
#im = im.transpose((2,0,1))
im = np.expand_dims(im, axis=0)
model = VGG16(weights="imagenet")
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
out_0 = model.predict(im)

out_old=out_0
i=0
for img in glob.glob(pathname):
    im=cv2.imread(im_test)
    im = cv2.resize(im, (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68

#im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    model = VGG16(weights="imagenet")
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im) 
    feat=np.concatenate((out, out_old), axis=0)
    out_old=feat
    i=i+1
    if i>20:
        break

X_train, X_test, y_train, y_test = train_test_split(feat, result['TARGET'][:feat.shape[0]], train_size=0.8, random_state=0)

#Train the model 
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(X_train, y_train)

#Prediction
y_pred = svr_rbf.predict(X_test)

# accuracy : mean square error
print("Mean Squared error: {}", mean_squared_error(y_test, y_pred))
print(y_pred[:10])
print(np.std(y_pred))