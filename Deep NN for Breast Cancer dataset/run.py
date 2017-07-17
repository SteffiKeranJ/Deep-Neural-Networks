import keras
import tensorflow as tf
from keras.models import Sequential
# Since all the layers are fully-connected, use Dense
from keras.layers import Dense
import numpy
numpy.random.seed(7)

# Load the Dataset "pima-indians-diabetes"
# The Dataset is Available at http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/
# First 8 columns consists of features 
# Last column is the binary label
ds = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# Feature Vector X
X = ds[:,0:8]
# Class Label Y
Y = ds[:,8]

# At first, let's use a 4-layer neural network
# Our model consists of Sequential Layers
model = Sequential()
# Add the first layer to the model
# First layer has 12 neurons and takes in 8 attributes
model.add(Dense(12,input_dim = 8,activation = 'relu'))
# Add the second layer to the model
# Second layer has 8 neurons and takes in 8 attributes
model.add(Dense(8,activation = 'relu'))
# Add the third layer to the model
model.add(Dense(8,activation = 'relu'))
#Add another layer
model.add(Dense(8,activation = 'relu'))

# Add the final and output layer 
# The output layer has just one neuron to predict the binary class label
model.add(Dense(1,activation = 'sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

# Train the model
model.fit(X,Y,epochs =250, batch_size = 10)

# Predict
prediction = model.predict(X)
result = [round(x[0]) for x in prediction]
print(result)
# Score
score = model.evaluate(X,Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
