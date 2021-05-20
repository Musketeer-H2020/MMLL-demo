# -*- coding: utf-8 -*-
'''
Neural Network model definition using Tensorflow Keras
'''

__author__ = "Marcos Fernández Díaz"
__date__ = "February 2021"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

# Define Keras model architecture
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Save the model architecture in JSON format
filename = 'keras_model_MLP_mnist.json'
model_json = model.to_json()
with open(filename, "w") as json_file:
    json_file.write(model_json)

print('Model architecture %s saved to disk' %filename)
model.summary()
