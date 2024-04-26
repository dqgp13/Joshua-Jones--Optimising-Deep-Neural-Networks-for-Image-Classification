# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:53:21 2024

@author: joshj
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Set the path to the dataset
dataset_path = r"C:\Users\joshj\OneDrive\Documents\Year 3\Project\Poster\Brain Tumours"

# Define the training and testing directories
train_dir = os.path.join(dataset_path, r"C:\Users\joshj\OneDrive\Documents\Year 3\Project\Poster\Brain Tumours\Training")
test_dir = os.path.join(dataset_path, r"C:\Users\joshj\OneDrive\Documents\Year 3\Project\Poster\Brain Tumours\Testing")

# Define the categories
categories = ["glioma", "meningioma", "notumor", "pituitary"]

# Load and preprocess the dataset
train_data = []
for category in categories:
    folder_path = os.path.join(train_dir, category)
    images = os.listdir(folder_path)
    count = len(images)
    train_data.append(pd.DataFrame({"Image": images, "Category": [category] * count, "Count": [count] * count}))

train_df = pd.concat(train_data, ignore_index=True)

# Set the image size
image_size = (150, 150)

# Set the batch size for training
batch_size = 32

# Set the number of epochs for training
epochs = 20

train_datagen = ImageDataGenerator(rescale=1./255)

X_train = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_datagen = ImageDataGenerator(rescale=1./255)

X_test = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'ReLU', input_shape=(image_size[0], image_size[1], 3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation = "ReLU"),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128, (3,3), activation = "ReLU"),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128, (3,3), activation = "ReLU"),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = "ReLU"),
    tf.keras.layers.Dense(4, activation = 'softmax')
    ])
                           
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train,
    steps_per_epoch=X_train.samples // batch_size,
    epochs=epochs,
    validation_data=X_test,
    validation_steps=X_test.samples // batch_size
)

training_loss = history.history['loss'] 
test_loss = history.history['val_loss']
training_accuracy = history.history['accuracy']
test_accuracy = history.history['val_accuracy']

epoch_range = range(1, epochs + 1)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training/Test Loss")
plt.plot(epoch_range, training_loss, label = "Training loss")
plt.plot(epoch_range, test_loss, color = "red", label = "Test loss")
plt.xticks([2,4,6,8,10,12,14,16,18,20])
plt.legend(loc = "upper right")
plt.show()

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training/Test Accuracy")
plt.plot(epoch_range, training_accuracy, label = "Training accuracy")
plt.plot(epoch_range, test_accuracy, color = "red", label = "Test accuracy")
plt.xticks([2,4,6,8,10,12,14,16,18,20])
plt.legend(loc = "center right")
plt.show()

loss, accuracy = model.evaluate(X_test, steps=X_test.samples // batch_size)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)