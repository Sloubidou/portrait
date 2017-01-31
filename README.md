# portrait
DataChallenge Portrait ML

Visualization of data
	>Exploring_Data.ipynb: ipython with data visualisation

Method 1 : Regression/classification on our built features
Extracting the features and building the new matrix of features
>angle_features.py : building the angle_features
	>color_features.py : building the color_features
	>quality_features.py : building the quality_features
	>spatial_features.py : building the spatial_features
>dataframe_features.py : code that builds our features matrix from the features below

	>dataframe.csv : new matrix from our features extraction
a) Prediction from features (directly)
>cross_val_direct_prediction.py : cross_validation on the features extracted (below)

>result_svm.csv : csv with the prediction for this method

b) Prediction from impacts predicted from features (indirectly)
	>prediction_impacts.py : algorithm to predict the impacts

	>essai_impact_test.csv : csv with the predicted impacts

Method 2 : Neural Network
> vgg.py: algorithm to perform the convolutional neural network

>... : csv with the predictions for this method
