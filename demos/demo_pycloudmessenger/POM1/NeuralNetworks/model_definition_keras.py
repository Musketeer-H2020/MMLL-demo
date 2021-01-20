# -*- coding: utf-8 -*-
'''
Neural Network model definition using Keras
'''

__author__ = "Marcos Fernández Díaz"
__date__ = "May 2019"

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

# Define Keras model architecture
model = Sequential()
model.add(Dense(15, input_shape=(111,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Save the model architecture in JSON format
filename = 'keras_model_MLP_income_raw.json'
model_json = model.to_json()
with open(filename, "w") as json_file:
    json_file.write(model_json)

print('Model architecture %s saved to disk' %filename)
model.summary()
