# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 11:37:42 2023

@author: joshj
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# Create a dataset with 10,000 samples.
X, y = make_circles(n_samples = 10000,
                    noise= 0.05,
                    random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

plt.scatter(X_train[:,0], X_train[:,1], marker = "o", s = 1, c = y_train,cmap="coolwarm_r")
plt.title("Training data")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
plt.scatter(X_test[:,0], X_test[:,1], marker = "o", s = 1, c = y_test,cmap="coolwarm_r")
plt.title("Test data")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

#Create neural network model.
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation = 'tanh'),
    tf.keras.layers.Dense(16, activation = 'tanh'),
    tf.keras.layers.Dense(1, activation = 'tanh')
    ])
predictions = model(X_train).numpy()
predictions = [0 if val < 0 else 1 for val in predictions]

eta = 1
sgd_optimizer = tf.keras.optimizers.SGD(learning_rate = eta)
model.compile(optimizer= sgd_optimizer,
              loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs = 50, validation_data=(X_test, y_test))
training_loss = history.history['loss'] 
test_loss = history.history['val_loss']
epochs = range(1, len(training_loss) + 1)

plt.plot(epochs, training_loss)
plt.plot(epochs, test_loss, color='red')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("η = %s" %(eta))
plt.show()




