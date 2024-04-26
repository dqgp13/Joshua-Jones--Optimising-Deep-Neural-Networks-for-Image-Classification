# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:06:33 2024

@author: joshj
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import block_reduce


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

#Display the original and edge-detected images
plt.imshow(image)
plt.axis('off')
plt.show()
plt.imshow(edge_detected_image)
plt.axis('off') 
plt.show()
plt.imshow(blurred_image)
plt.axis('off')  
plt.show()
plt.imshow(sharpened_image)
plt.axis('off')  
plt.show()

#mean_pool=block_reduce(edge_detected_image, block_size=(2,2,1), func=np.mean)
max_pool_identity=block_reduce(image, block_size=(2,2,1), func=np.max)
max_pool_edge_detected=block_reduce(edge_detected_image, block_size=(2,2,1), func=np.max)
max_pool_blurred=block_reduce(blurred_image, block_size=(2,2,1), func=np.max)
max_pool_sharpened=block_reduce(sharpened_image, block_size=(2,2,1), func=np.max)

# plt.imshow(mean_pool)
# plt.axis('off')  
# plt.show()

plt.imshow(max_pool_identity)
plt.axis('off')  
plt.show()
print(image.shape)
print(max_pool_identity.shape)

plt.imshow(max_pool_edge_detected)
plt.axis('off')  
plt.show()
print(edge_detected_image.shape)
print(max_pool_edge_detected.shape)

plt.imshow(max_pool_blurred)
plt.axis('off')  
plt.show()
print(blurred_image.shape)
print(max_pool_blurred.shape)

plt.imshow(max_pool_sharpened)
plt.axis('off')  
plt.show()
print(sharpened_image.shape)
print(max_pool_sharpened.shape)
