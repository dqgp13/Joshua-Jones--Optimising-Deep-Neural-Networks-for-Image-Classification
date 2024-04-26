# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:06:33 2024

@author: joshj
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import MaxPooling2D

# Read the image
image = cv2.imread(r"C:\Users\joshj\OneDrive\Documents\Year 3\Project\Poster\No tumour example.jpg")

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

# Display the original and edge-detected images
plt.set_cmap("gray")
plt.imshow(image)
plt.axis('off')  # Turn off axis
plt.show()
plt.imshow(edge_detected_image, cmap='gray')
plt.axis('off')  # Turn off axis
plt.show()
plt.imshow(blurred_image)
plt.axis('off')  # Turn off axis
plt.show()
plt.imshow(sharpened_image)
plt.axis('off')  # Turn off axis
plt.show()