# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:06:08 2024

@author: joshj
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import MaxPooling2D
from skimage.measure import block_reduce

# Read the image
image = cv2.imread(r"C:\Users\joshj\OneDrive\Documents\Year 3\Project\Essay\Fourier transform\Durham Cathedral Image\Durham Cathedral Image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Convert image colour to RGB from BGR

# Define the 3x3 edge detection kernel
edge_detection_kernel = np.array([[-1, -1, -1],
                                  [-1,  8, -1],
                                  [-1, -1, -1]])
# Define the 3x3 blurring kernel
blurring_kernel = np.array([[1/9, 1/9, 1/9],
                            [1/9,  1/9, 1/9],
                            [1/9, 1/9, 1/9]])
# Define the 3x3 sharpening kernel
sharpening_kernel = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])

# Apply the kernels to the image
edge_detected_image = cv2.filter2D(image, -1, edge_detection_kernel)
blurred_image = cv2.filter2D(image, -1, blurring_kernel)
sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

fig, ax = plt.subplots(figsize=(sharpened_image.shape[1]/100, sharpened_image.shape[0]/100), dpi=100)
plt.imshow(sharpened_image)
plt.axis('off')  # Turn off axis
plt.show()

max_pool_sharpened=block_reduce(sharpened_image, block_size=(2,2,1), func=np.max)
fig, ax = plt.subplots(figsize=(max_pool_sharpened.shape[1]/100, sharpened_image.shape[0]/100), dpi=100)
plt.imshow(max_pool_sharpened)
plt.axis('off')  
plt.show()
print(max_pool_sharpened.shape)

