# Imported libraries
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import numpy as np
import cv2
import PIL
import io
import html
import time
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import random
from os import listdir
from os.path import isfile, join

from math import floor
from scipy.ndimage.interpolation import zoom, rotate

import imageio
import face_recognition

# Creating a Dictionary to define Image Dimensions
image_dimensions = {'height':256, 'width':256, 'channels':3}

# Create a Classifier class

class Classifier:
    def __init__():
        self.model = 0
    
    def predict(self, x):
        return self.model.predict(x)
    
    def fit(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        self.model.load_weights(path)

# Create a MesoNet class using the Classifier

class Meso4(Classifier):
  # Created init method
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        # Defined a gradient Descent optimizer variable and set the LR in the constructor
        optimizer = Adam(lr = learning_rate)
        # Defined parameters to compile the model
        self.model.compile(optimizer = optimizer,
                           loss = 'mean_squared_error',
                           metrics = ['accuracy'])
    # Create method called init_model
    def init_model(self): 
      #Create a variable X and assigned input layer to pass the image dimensions
        x = Input(shape = (image_dimensions['height'],
                           image_dimensions['width'],
                           image_dimensions['channels']))
        # Convoulutional Layers
        x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
        # Batch Normalization
        x1 = BatchNormalization()(x1)
        # Max Pooling 
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return Model(inputs = x, outputs = y)