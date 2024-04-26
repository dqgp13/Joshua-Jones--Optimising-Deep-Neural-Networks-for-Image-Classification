# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:46:02 2024

@author: joshj
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-4,4,0.005)

def ReLU(x):
    return max(0, x)
def LReLU(x, k):
    return max(k*x, x)

y = np.array([ReLU(xi) for xi in x])
k = 0.1
z = np.array([LReLU(xi, k) for xi in x])

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.title("ReLU function")
plt.grid()
plt.show()

plt.plot(x, z)
plt.xlabel("x")
plt.ylabel("LReLU(x)")
plt.yticks([-0.5,0,0.5,1,1.5,2.0,2.5,3.0,3.5,4.0])
plt.title("Leaky ReLU function")
plt.grid()
plt.show()