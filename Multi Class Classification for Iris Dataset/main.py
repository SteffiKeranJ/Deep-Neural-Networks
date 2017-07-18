import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
# random seed for reproducibility
seed = 7
np.random.seed(seed)
# load dataset
dframe = pd.read_csv("irisdata.csv", header=None)
ds = dframe.values
# Input X
X = ds[:,0:4].astype(float)
# Input Y
Y = ds[:,4]
# encode class values as integers
# Fit, transform and convert to One Hot values
encoder = LabelEncoder()
encoder.fit(Y)
encY = encoder.transform(Y)
# convert integers to  one hot encoded variables 
onehotY = np_utils.to_categorical(encY)
# baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(4, input_dim=4, activation="relu", kernel_initializer="normal"))
	model.add(Dense(3, activation="sigmoid", kernel_initializer="normal"))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, onehotY, test_size=0.33, random_state=seed)
estimator.fit(X_train, Y_train)
prediction = estimator.predict(X_test)
print(prediction)
print(encoder.inverse_transform(prediction))