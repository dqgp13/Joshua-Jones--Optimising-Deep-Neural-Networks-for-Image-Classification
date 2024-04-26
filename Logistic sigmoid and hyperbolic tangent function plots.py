# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:34:29 2023

@author: joshj
"""

import matplotlib.pyplot as plt
import numpy as np
import math

x = np.arange(-8,8,0.01)
def sigmoid(x):
    return (1/(1+math.exp(-x)))
def tanh(x):
    return ((math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x)))

y = np.array([sigmoid(xi) for xi in x])
z = np.array([tanh(xi) for xi in x])

plt.figure(0)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("σ (x)")

plt.figure(1)
plt.plot(x, z)
plt.xlabel("x")
plt.ylabel("tanh(x)")