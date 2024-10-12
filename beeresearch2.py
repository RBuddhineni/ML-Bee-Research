# -*- coding: utf-8 -*-
# @title
project = "bees"

import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from IPython import display
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from tensorflow.image import resize_with_pad, ResizeMethod
from sklearn.model_selection import train_test_split
from PIL import Image
import requests
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from sklearn.metrics import accuracy_score

dataset_url_prefix_dict = {

    #TensorFlow dataset stored on Google API by AI Scholars program
    "bees"      : "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Safeguarding%20Bee%20Health/"
}

wget_command = f'wget -q --show-progress "{dataset_url_prefix_dict[project]}'
!{wget_command + 'images.npy" '}
!{wget_command + 'labels.npy" '}

images = np.load("images.npy")
labels = np.load("labels.npy")

!rm images.npy labels.npy

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Conv3D, Flatten

# Using the get_dummies() function to one-hot encode labels.
labels_ohe = np.array(pd.get_dummies(labels))

# Select your feature (X) and labels (y).
y = labels_ohe
X = images

# Splits data into training and testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# Initialize your model
cnn_model = Sequential()


# Input layer
cnn_model.add(Input(shape=X_train.shape[1:]))

# First layer
cnn_model.add(Conv2D(32,(3,3), activation='relu', padding="same"))
cnn_model.add(MaxPooling2D((2, 2)))

# Second layer
cnn_model.add(Conv2D(16,(3,3), activation='relu', padding="same"))
cnn_model.add(MaxPooling2D((2, 2)))

# Third layer
cnn_model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
cnn_model.add(MaxPooling2D((2, 2)))

# Fourth layer
cnn_model.add(Conv2D(16,(3,3), activation='relu', padding="same"))
cnn_model.add(MaxPooling2D((2, 2)))

# Fifth layer
cnn_model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
cnn_model.add(MaxPooling2D((2, 2)))


# Flattening layer
cnn_model.add(Flatten())

# Hidden (dense) layer with 64 nodes, and relu activation function.
cnn_model.add(Dense(64, activation='relu'))

# Final output layer that uses a softmax activation function.
cnn_model.add(Dense(len(set(labels)), activation='softmax'))

# Compile
metrics_to_track = ['categorical_crossentropy', 'accuracy']
cnn_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=metrics_to_track)

one_hot_encoding_to_label_dict = {np.argmax(ohe):label for ohe, label in zip(labels_ohe, labels)}

# This function takes in a vector, and outputs the label.
def ScoreVectorToPredictions(prob_vector):
  class_num = np.argmax(prob_vector) # Finds which element in the vector has the highest score.
  class_name = one_hot_encoding_to_label_dict[class_num] # Figures out the label that corresponds to this element.
  return class_name, max(prob_vector) # Returns the label as well as the probabilty that the model assigned to this prediction.

for i in range(3):
  print(ScoreVectorToPredictions(y_train[i]))
  print(ScoreVectorToPredictions(y_test[i]))

cnn_model.summary()

cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

#-----------------------------

#Data Augmentation
from numpy import flipud

## A function to create an augmented image from an original.
def createAugmentedImage(original_image):
  new_image = flipud(original_image)
  return new_image

# Transforms the first image of the dataset
new_image = createAugmentedImage(X_train[0])

f, ax = plt.subplots(ncols=2)
ax[0].imshow(X_train[0])
ax[0].set_title('New Image')
ax[1].imshow(new_image)
ax[1].set_title('Augmented Image')
plt.show()

for i in range(100):
  new_X = createAugmentedImage(X_train[i])
  new_y = y_train[i]

  if i == 0:
    X_train_augment = [new_X]
    y_train_augment = [new_y]
  else:
    X_train_augment = np.append(X_train_augment, [new_X], axis=0)
    y_train_augment = np.append(y_train_augment, [new_y], axis=0)


print("Dimensions of augmented X:", X_train_augment.shape)
print("Dimensions of y:", y_train_augment.shape)

cnn_model.fit(X_train_augment, y_train_augment, validation_data=(X_test, y_test), epochs=5)

one_hot_encoding_to_label_dict = {np.argmax(ohe):label for ohe, label in zip(labels_ohe, labels)}
def ScoreVectorToPredictions(prob_vector):
  class_num = np.argmax(prob_vector) # Finds which element in the vector has the highest score.
  class_name = one_hot_encoding_to_label_dict[class_num] # Figures out the label that corresponds to this element.
  return class_name, max(prob_vector) # Returns the label as well as the probabilty that the model assigned to this prediction.

# Predicts on the first three images from the test dataset
# (you could predict on all of the samples, just doing 3 for speed)
scores = cnn_model.predict(X_test[:3])
print('scores: ', scores[0])

class_name, prob = ScoreVectorToPredictions(scores[0]) # Gets the model predictions and associated probabilitie
true_label, true_prob = ScoreVectorToPredictions(y_test[0]) # Gets the true labels

print('model prediction: %s (%.02f probability)' % (class_name, prob))
print('true label: %s (%.02f probability)' % (true_label, true_prob))

plt.figure()
plt.imshow(X_test[2]) # Number is interchangeable
plt.show()

#----------------------------

#Transfer Learning
from keras.applications import MobileNetV2, VGG16
mobile_net = VGG16(include_top=True)
mobile_net.summary()

from keras import Model

def new_output_layer(input_layer):
  return Dense(len(set(labels)), activation='softmax')(input_layer)

output = new_output_layer(mobile_net.layers[-2].output)
input = mobile_net.input
transfer_cnn = Model(input, output)

# print the summary
transfer_cnn.summary()

for layer in transfer_cnn.layers:
    layer.trainable = False

transfer_cnn.layers[-1].trainable = True

transfer_cnn.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', 'categorical_crossentropy'])

transfer_cnn.summary()

# Takes in an image, a new height, and a new width
# and resizes the image, plus converts from greyscale to 3 RGB color channels.
def ResizeImages(images, height, width):
  return np.array([resize_with_pad(image, height, width, antialias=True) for image in images]).astype(int)

# Resize Image to match dimensions
X_train_resized = ResizeImages(X_train, 224, 224)
X_test_resized = ResizeImages(X_test, 224, 224)

print("Dim X_train_resized:", X_train_resized.shape)
print("Dim X_test_resized:", X_test_resized.shape)

transfer_cnn.fit(X_train_resized, y_train, validation_data=(X_test_resized, y_test), epochs=3)

one_hot_encoding_to_label_dict = {np.argmax(ohe):label for ohe, label in zip(labels_ohe, labels)}
def ScoreVectorToPredictions(prob_vector):
  class_num = np.argmax(prob_vector) # Finds which element in the vector has the highest score.
  class_name = one_hot_encoding_to_label_dict[class_num] # Figures out the label that corresponds to this element.
  return class_name, max(prob_vector) # Returns the label as well as the probabilty that the model assigned to this prediction.

# Predicts on the first three images from the test dataset
# (you could predict on all of the samples, just doing 3 for speed)
scores = transfer_cnn.predict(X_test_resized[:3])
print('scores: ', scores[0])

class_name, prob = ScoreVectorToPredictions(scores[0]) # Gets the model predictions and associated probabilitie
true_label, true_prob = ScoreVectorToPredictions(y_test[0]) # Gets the true labels

print('model prediction: %s (%.02f probability)' % (class_name, prob))
print('true label: %s (%.02f probability)' % (true_label, true_prob))


plt.figure()
plt.imshow(X_test_resized[2]) # Looking at the first "color channel"
plt.show()

# Upload a file of a healthy/unhealthy bee and the program will predict whether the bee has varroa mite or not
from google.colab import files
from tensorflow.keras.preprocessing import image
import numpy as np

# Upload the image file
uploaded = files.upload()

# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    # Load the image with the target size (224, 224 for VGG16)
    img = image.load_img(img_path, target_size=(224, 224))
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Expand dimensions to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image array
    img_array /= 255.0
    return img_array

# Function to predict if the bee is healthy or not
def predict_bee_health(img_path):
    # Load and preprocess the image
    img = load_and_preprocess_image(img_path)
    # Make prediction
    predictions = transfer_cnn.predict(img)
    class_name, prob = ScoreVectorToPredictions(predictions[0])  # Get label and probability
    return class_name, prob

# Example usage
for uploaded_file in uploaded.keys():
    # Predict using the uploaded image file
    class_name, probability = predict_bee_health(uploaded_file)
    print('Model Prediction for %s: %s (%.2f probability)' % (uploaded_file, class_name, probability))
