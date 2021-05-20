# -*- coding: utf-8 -*-
'''
Neural Network model definition using Tensorflow Keras
'''

__author__ = "Marcos Fernández Díaz"
__date__ = "February 2021"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation

# Define Keras model architecture
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
# Save the model architecture in JSON format
filename = 'keras_model_MLP_mnist_tanh.json'
model_json = model.to_json()
with open(filename, "w") as json_file:
    json_file.write(model_json)
print('Model architecture %s saved to disk' %filename)


"""
# CNNs
IMG_SIZE=28 # MNIST images

model = Sequential()
model.add(Conv2D(8, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))

# Save the model architecture in JSON format
filename = 'keras_model_CNN_mnist.json'
model_json = model.to_json()
with open(filename, "w") as json_file:
    json_file.write(model_json)

print('Model architecture %s saved to disk' %filename)
model.summary()
"""
