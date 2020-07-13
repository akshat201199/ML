# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 12:37:50 2020

@author: Akshat Shah
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
        'TRAINING SET PATH',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_set = validation_datagen.flow_from_directory(
        'VALIDATION SET PATH',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2 , strides = 2))
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2,strides = 2))
cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units = 128,activation = 'relu'))
cnn.add(tf.keras.layers.Dense(units = 1,activation = 'sigmoid'))

cnn.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(x = training_set, validation_data = validation_set, epochs = 25)

test_image = image.load_img('TEST IMAGE PATH',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
if result[0][0] == 1:
    prediction = "ENTER 1st CLASS"
else:
    prediction = "ENTER 2nd CLASS"
print(prediction)    
    

