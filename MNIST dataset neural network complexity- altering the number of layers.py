# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:57:26 2024

@author: joshj
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 20:14:34 2024

@author: joshj
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import keras

from keras.datasets import mnist
from keras.utils import to_categorical

mnist_data = mnist.load_data()
(X_train, y_train), (X_test, y_test) = mnist_data

X_train = X_train.reshape((X_train.shape[0], 28*28)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28*28)).astype('float32')
X_train = X_train/255
X_test = X_test/255

# Convert y_train into one-hot format
temp = []
for i in range(len(y_train)):
    temp.append(to_categorical(y_train[i], num_classes=10))
y_train = np.array(temp)
# Convert y_test into one-hot format
temp = []
for i in range(len(y_test)):    
    temp.append(to_categorical(y_test[i], num_classes=10))
y_test = np.array(temp)

# print(y_train.shape)
# print(y_train[0])

colours = ['navy', 'blue','deepskyblue', 'limegreen', 'orange',  'red', 'darkviolet', 'fuchsia']

for i in range(1,6):
    #Create neural network model.
    if i == 1:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(8, activation = 'ReLU'),
            tf.keras.layers.Dense(10, activation = 'softmax')
            ])
    elif i == 2:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(8, activation = 'ReLU'),
            tf.keras.layers.Dense(8, activation = 'ReLU'),
            tf.keras.layers.Dense(10, activation = 'softmax')
            ])
    elif i == 3:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(8, activation = 'ReLU'),
            tf.keras.layers.Dense(8, activation = 'ReLU'),
            tf.keras.layers.Dense(8, activation = 'ReLU'),
            tf.keras.layers.Dense(10, activation = 'softmax')
            ])
    elif i == 4:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(8, activation = 'ReLU'),
            tf.keras.layers.Dense(8, activation = 'ReLU'),
            tf.keras.layers.Dense(8, activation = 'ReLU'),
            tf.keras.layers.Dense(8, activation = 'ReLU'),
            tf.keras.layers.Dense(10, activation = 'softmax')
            ])
    elif i == 5:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(8, activation = 'ReLU'),
            tf.keras.layers.Dense(8, activation = 'ReLU'),
            tf.keras.layers.Dense(8, activation = 'ReLU'),
            tf.keras.layers.Dense(8, activation = 'ReLU'),
            tf.keras.layers.Dense(8, activation = 'ReLU'),
            tf.keras.layers.Dense(10, activation = 'softmax')
            ])
    
    predictions = model(X_train).numpy()
    #print(predictions[0])
    predictions = np.argmax(predictions, axis=1)
    #print(predictions[0])

    model.compile(optimizer= 'adam',
              loss='mean_squared_error',
              metrics = ['accuracy'])

    history = model.fit(X_train, y_train, epochs = 25, validation_data=(X_test, y_test))
    training_loss = history.history['loss'] 
    test_loss = history.history['val_loss']
    training_accuracy = history.history['accuracy']
    test_accuracy = history.history['val_accuracy']

    epochs = range(1, len(training_loss) + 1)
    n_layers = i
    
    plt.figure(0)
    if i == 1:
        plt.plot(epochs, training_loss, color=colours[i], label = ("%s layer" %(n_layers)))
    else:
        plt.plot(epochs, training_loss, color=colours[i], label = ("%s layers" %(n_layers)))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.legend(loc="upper right")
    plt.figure(1)
    plt.plot(epochs, test_loss, color=colours[i])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Test loss")
    plt.figure(2)
    plt.plot(epochs, training_accuracy, color=colours[i])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training accuracy")
    plt.figure(3)
    plt.plot(epochs, test_accuracy, color=colours[i])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Test accuracy")
    
    keras.backend.clear_session()