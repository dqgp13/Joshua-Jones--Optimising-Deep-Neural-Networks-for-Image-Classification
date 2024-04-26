# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 20:15:59 2023

@author: joshj
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import math

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# Create a dataset with 10,000 samples.
X, y = make_circles(n_samples = 10000,
                    noise= 0.05,
                    random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# plt.scatter(X_train[:,0], X_train[:,1], marker = "o", s = 1, c = y_train,cmap="coolwarm_r")
# plt.title("Training data")
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.show()
# plt.scatter(X_test[:,0], X_test[:,1], marker = "o", s = 1, c = y_test,cmap="coolwarm_r")
# plt.title("Test data")
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.show()

#Create neural network model.
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation = 'tanh'),
    tf.keras.layers.Dense(16, activation = 'tanh'),
    tf.keras.layers.Dense(1, activation = 'tanh')
    ])
predictions1 = model1(X_train).numpy()
predictions1 = [0 if val < 0 else 1 for val in predictions1]

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation = 'tanh'),
    tf.keras.layers.Dense(16, activation = 'tanh'),
    tf.keras.layers.Dense(1, activation = 'tanh')
    ])
predictions2 = model2(X_train).numpy()
predictions2 = [0 if val < 0 else 1 for val in predictions2]

model3 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation = 'tanh'),
    tf.keras.layers.Dense(16, activation = 'tanh'),
    tf.keras.layers.Dense(1, activation = 'tanh')
    ])
predictions3 = model3(X_train).numpy()
predictions3 = [0 if val < 0 else 1 for val in predictions3]

model4 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation = 'tanh'),
    tf.keras.layers.Dense(16, activation = 'tanh'),
    tf.keras.layers.Dense(1, activation = 'tanh')
    ])
predictions4 = model4(X_train).numpy()
predictions4 = [0 if val < 0 else 1 for val in predictions4]


initial_eta = 0.1
decay_rate = 0.1
def polynomial_decay_power_one(epoch):
    #For initial learning rate 0.1 and final learning rate 0.01, set the decay rate to 0.18.
    return initial_eta/(1 + 0.18*epoch)
def polynomial_decay_power_half(epoch):
    #For initial learning rate 0.1 and final learning rate 0.01, set the decay rate to 1.272792206.
    return initial_eta/(1 + 1.272792206*epoch**0.5)
def exponential_decay(epoch):
    #For initial learning rate 0.1 and final learning rate 0.01, set the decay rate to -0.04605170186.
    return initial_eta*math.exp(-0.04605170186*epoch)
def step_decay(epoch):
    if epoch < 10:
        lr = 0.1
    elif epoch < 20:
        lr = 0.0775
    elif epoch < 30:
        lr = 0.055
    elif epoch < 40:
        lr = 0.0325
    else:
        lr = 0.01
    return lr

lr_exp_decay = tf.keras.callbacks.LearningRateScheduler(exponential_decay, verbose = 1)
lr_power_one_decay = tf.keras.callbacks.LearningRateScheduler(polynomial_decay_power_one, verbose = 1)
lr_power_half_decay = tf.keras.callbacks.LearningRateScheduler(polynomial_decay_power_half, verbose = 1)
lr_step_decay = tf.keras.callbacks.LearningRateScheduler(step_decay, verbose = 1)

sgd_optimizer1 = tf.keras.optimizers.SGD(learning_rate = initial_eta)
sgd_optimizer2 = tf.keras.optimizers.SGD(learning_rate = initial_eta)
sgd_optimizer3 = tf.keras.optimizers.SGD(learning_rate = initial_eta)
sgd_optimizer4 = tf.keras.optimizers.SGD(learning_rate = initial_eta)

model1.compile(optimizer= sgd_optimizer1,
              loss='mean_squared_error')
model2.compile(optimizer= sgd_optimizer2,
              loss='mean_squared_error')
model3.compile(optimizer= sgd_optimizer3,
              loss='mean_squared_error')
model4.compile(optimizer= sgd_optimizer4,
              loss='mean_squared_error')

history_exp_decay = model1.fit(X_train, y_train, epochs = 50, validation_data=(X_test, y_test), 
                              callbacks = [lr_exp_decay])
history_power_one_decay = model2.fit(X_train, y_train, epochs = 50, validation_data=(X_test, y_test), 
                                    callbacks = [lr_power_one_decay])
history_power_half_decay = model3.fit(X_train, y_train, epochs = 50, validation_data=(X_test, y_test), 
                                     callbacks = [lr_power_half_decay])
history_step_decay = model4.fit(X_train, y_train, epochs = 50, validation_data=(X_test, y_test), 
                                     callbacks = [lr_step_decay])

power_one_training_loss = history_power_one_decay.history['loss'] 
power_one_test_loss = history_power_one_decay.history['val_loss']
power_half_training_loss = history_power_half_decay.history['loss'] 
power_half_test_loss = history_power_half_decay.history['val_loss']
exp_training_loss = history_exp_decay.history['loss'] 
exp_test_loss = history_exp_decay.history['val_loss']
step_training_loss = history_step_decay.history['loss'] 
step_test_loss = history_step_decay.history['val_loss']

epochs = range(1, len(power_one_training_loss) + 1)

plt.plot(epochs, power_one_training_loss)
plt.plot(epochs, power_one_test_loss, color='red')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Power one")
plt.show()

plt.plot(epochs, power_half_training_loss)
plt.plot(epochs, power_half_test_loss, color='red')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Power half")
plt.show()

plt.plot(epochs, exp_training_loss)
plt.plot(epochs, exp_test_loss, color='red')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Exponential")
plt.show()

plt.plot(epochs, step_training_loss)
plt.plot(epochs, step_test_loss, color='red')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Step")
plt.show()

epoch_array = np.arange(0, len(power_one_training_loss) + 1)

power_one_decay = [polynomial_decay_power_one(i) for i in epoch_array]
power_half_decay = [polynomial_decay_power_half(i) for i in epoch_array]
exp_decay = [exponential_decay(i) for i in epoch_array]
step_decay = [step_decay(i) for i in epoch_array]

plt.plot(epoch_array, power_one_decay, label = "Power one")
plt.plot(epoch_array, power_half_decay, color = 'red', label = "Power half")
plt.plot(epoch_array, exp_decay, color = 'orange', label = "Exponential")
plt.plot(epoch_array, step_decay, color = 'green', label = "Step")
plt.legend(loc="upper right")
plt.xlabel("Epochs")
plt.ylabel("Learning rate")
plt.title("Learning rate decay")
plt.show()


