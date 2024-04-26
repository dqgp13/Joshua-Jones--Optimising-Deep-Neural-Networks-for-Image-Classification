# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:41:16 2024

@author: joshj
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import MaxPooling2D

# Read the image
image = cv2.imread(r"C:\Users\joshj\OneDrive\Documents\Year 3\Project\Poster\No tumour example.jpg")

def calculate_2dft(input):
    return np.fft.fft2(input)

grayscale_image = image[:, :, :3].mean(axis=2)  #Average the three channels to convert the image to grayscale
ft_image = calculate_2dft(grayscale_image)

plt.set_cmap("gray")

plt.imshow(image)
plt.axis('off')  # Turn off axis
plt.show()
plt.imshow(abs(ft_image))
plt.axis('off')  # Turn off axis
plt.show()
plt.imshow(np.log(abs(ft_image)))
plt.axis('off')  # Turn off axis
plt.show()

kernel_shape = image.shape
height = kernel_shape[0]
width = kernel_shape[1]
print(kernel_shape)
# print(height)
# print(width)

#Intialise the kernel to be the same size as the image
kernel = np.zeros(shape = (height,width))
#Then chnage the top left of the kernel to be the edge detection kernel
kernel[0,0] = kernel[1,0] = kernel[2,0] = -1
kernel[0,1] = kernel[2,1] = -1
kernel[0,2] = kernel[1,2] = kernel[2,2] = -1
kernel[1,1] = 8
print(kernel)

ft_kernel = calculate_2dft(kernel)

plt.imshow(abs(ft_kernel))
plt.axis('off')  # Turn off axis
plt.show()
plt.imshow(np.log(abs(ft_kernel)))
plt.axis('off')  # Turn off axis
plt.show()

ft_product = np.multiply(ft_image, ft_kernel)
plt.imshow(abs(ft_product))
plt.axis('off')  # Turn off axis
plt.show()
plt.imshow(np.log(abs(ft_product)))
plt.axis('off')  # Turn off axis
plt.show()

def calculate_2dift(input):
    return np.fft.ifft2(input)

ft_inverse = calculate_2dift(ft_product)
plt.imshow(abs(ft_inverse))
plt.axis('off')  # Turn off axis
plt.show()
