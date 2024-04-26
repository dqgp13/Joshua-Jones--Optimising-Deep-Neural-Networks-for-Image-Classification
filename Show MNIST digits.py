# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:40:06 2024

@author: joshj
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from keras.datasets import mnist

mnist_data = mnist.load_data()
(X_train, y_train), (X_test, y_test) = mnist_data

#Find indices for the digits 0 to 9.
for i in range(50):
    print("i is ", i, "MNIST digit is ", y_train[i])


index_vals = [21,40,28,27,9,35,18,15,17,43]
figure, axes = plt.subplots(nrows = 2,ncols = 5, sharex=False, 
    sharey=True, figsize=(12, 5.7))
for i in range(0,10):
    ax = axes[i//5, i%5]
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("%s" %y_train[index_vals[i]], fontsize = 22)
    ax.imshow(X_train[index_vals[i]], cmap='gray')
    
plt.tight_layout()
plt.show()