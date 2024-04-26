# # -*- coding: utf-8 -*-
# """
# Created on Tue Jan  2 20:14:34 2024

# @author: joshj
# """

import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import keras
import time

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
time_values = []

for i in range(0,8):
    #Create neural network model.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(2**i, activation = 'ReLU'),
        tf.keras.layers.Dense(10, activation = 'softmax')
        ])
    predictions = model(X_train).numpy()
    #print(predictions[0])
    predictions = np.argmax(predictions, axis=1)
    #print(predictions[0])

    model.compile(optimizer= 'adam',
              loss='mean_squared_error',
              metrics = ['accuracy'])
    
    t0 = time.time()
    history = model.fit(X_train, y_train, epochs = 25, validation_data=(X_test, y_test))
    t1 = time.time()
    time_taken = t1 - t0
    time_values.append(time_taken)
    training_loss = history.history['loss'] 
    test_loss = history.history['val_loss']
    training_accuracy = history.history['accuracy']
    test_accuracy = history.history['val_accuracy']

    epochs = range(1, len(training_loss) + 1)
    n_nodes = 2**i
    
    plt.figure(0)
    plt.plot(epochs, training_loss, color=colours[i], label = "%s" %n_nodes)
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.legend(loc="upper right")
    plt.figure(1)
    plt.plot(epochs, test_loss, color=colours[i])
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Test loss")
    plt.figure(2)
    plt.plot(epochs, training_accuracy, color=colours[i])
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training accuracy")
    plt.figure(3)
    plt.plot(epochs, test_accuracy, color=colours[i])
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Test accuracy")
    
    keras.backend.clear_session()

log2_nodes = np.arange(0,8)
plt.figure(4)
plt.plot(log2_nodes, time_values)
plt.xlabel("$log_{2}$(number of nodes)")
plt.ylabel("Time (seconds)")
plt.title("Time taken to train network")

#Create a function to calculate number of parameters in our two layer network.
def n_nodes(n):
    return 795*n + 10

n_parameters = [n_nodes(2**i) for i in log2_nodes]
plt.figure(5)
plt.plot(log2_nodes, n_parameters)
plt.xlabel("$log_{2}$(number of nodes)")
plt.ylabel("Number of parameters")
plt.title("Number of parameters to train in the network")
