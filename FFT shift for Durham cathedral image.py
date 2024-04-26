# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:40:52 2024

@author: joshj
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import MaxPooling2D

# Read the image
image = cv2.imread(r"C:\Users\joshj\OneDrive\Documents\Year 3\Project\Essay\Fourier transform\Durham Cathedral Image\Durham Cathedral Image.jpg")
print(image.shape)
RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Convert image colour to RGB from BGR

def calculate_2dft(input):
    return np.fft.fft2(input)

image_red = RGB_image[:,:,0]
image_green = RGB_image[:,:,1]
image_blue = RGB_image[:,:,2]

ft_image_red = calculate_2dft(image_red)
ft_image_green = calculate_2dft(image_green)
ft_image_blue = calculate_2dft(image_blue)

ft_image = np.dstack([ft_image_red, ft_image_green, ft_image_blue])

height = image.shape[0]
width = image.shape[1]

#Intialise the kernel to be the same size as the image
kernel = np.zeros(shape = (height,width))

#Sharpening kernel
kernel[0,0]=kernel[0,2]=kernel[2,0]=kernel[2,2]=0
kernel[0,1]=kernel[1,0]=kernel[1,2]=kernel[2,1]=-1
kernel[1,1]=5

ft_kernel = calculate_2dft(kernel)

ft_product_red = np.multiply(ft_image_red, ft_kernel)
ft_product_green = np.multiply(ft_image_green, ft_kernel)
ft_product_blue = np.multiply(ft_image_blue, ft_kernel)

shifted_ft_product_red = np.fft.ifftshift(ft_product_red)
shifted_ft_product_green = np.fft.ifftshift(ft_product_green)
shifted_ft_product_blue = np.fft.ifftshift(ft_product_blue)

pool_size = 1
min_width = int(np.floor(width*(0.5 - (pool_size/2))))
max_width = int(np.floor(width*(0.5 + (pool_size/2))))
min_height = int(np.floor(height*(0.5 - (pool_size/2))))
max_height = int(np.floor(height*(0.5 + (pool_size/2))))
print(min_height)

cropped_ft_product_red = shifted_ft_product_red[min_height:max_height, min_width:max_width]
cropped_ft_product_green = shifted_ft_product_green[min_height:max_height, min_width:max_width]
cropped_ft_product_blue = shifted_ft_product_blue[min_height:max_height, min_width:max_width]

ft_product = np.dstack([cropped_ft_product_red, cropped_ft_product_green, cropped_ft_product_blue])


def calculate_2dift(input):
    return np.fft.ifft2(input).real
#Although the imaginary part shoudl be 0, floating point errors can occur and result in 
#a small imaginary part, so we must return the real part.

shifted_ft_inverse_red = np.fft.fftshift(cropped_ft_product_red)
shifted_ft_inverse_green = np.fft.fftshift(cropped_ft_product_green)
shifted_ft_inverse_blue = np.fft.fftshift(cropped_ft_product_blue)

ft_inverse_red = calculate_2dift(shifted_ft_inverse_red)
ft_inverse_green = calculate_2dift(shifted_ft_inverse_green)
ft_inverse_blue = calculate_2dift(shifted_ft_inverse_blue)

ft_inverse = np.dstack([ft_inverse_red, ft_inverse_green, ft_inverse_blue])

fig, ax = plt.subplots(figsize=(RGB_image.shape[1]/100, RGB_image.shape[0]/100), dpi=100)
ax.imshow(RGB_image/255)
ax.axis('off')  # Turn off axis
plt.show()

fig, ax = plt.subplots(figsize=(ft_inverse.shape[1]/100, ft_inverse.shape[0]/100), dpi=100)
ax.imshow(ft_inverse/255)
ax.axis('off')  # Turn off axis
plt.show()