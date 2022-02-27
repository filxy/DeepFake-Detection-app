import streamlit as st
import cv2
import numpy as np
from PIL import Image
# Imported libraries
from IPython.display import display, Javascript, Image
from base64 import b64decode, b64encode
import numpy as np
import cv2
import PIL
import io
import html
import time
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
import random
import os
from os import listdir
from os.path import isfile, join

#from math import floor
#from scipy.ndimage.interpolation import zoom, rotate
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

st.header("Deep Fake Detector")

@st.cache
def load_image(uploaded_file):
  img = Image.open(uploaded_file)
  return img

def save_uploadedfile(uploaded_file):
  with open(os.path.join("https://github.com/filxy/test/tree/main/tempDir",uploaded_file.name),"wb") as f:
    f.write(uploaded_file.getbuffer())
    return st.success("Saved File:{} to tempDir".format(uploaded_file.name))

def main():
  Menu = ["image","About"]
  choice = st.sidebar.selectbox("Menu",Menu)
  if choice == "image prediction":
    st.subheader("Predict deep fake images")
    uploaded_file = st.file_uploader("Choose a image ")
    for uploaded_file in uploaded_file:
      bytes_data = uploaded_file.read()
      st.write("filename:", uploaded_file.name)
      st.write(bytes_data)
    else:
      st.write("about app")
if __name__ == '__main__':
        main()
uploaded_file = st.file_uploader("Choose a image file", type = ['jpeg','jpg'])

if uploaded_file is not None:
  # Convert the file to an opencv image.
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes, 1)
  opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
  resized = cv2.resize(opencv_image,(256,256))
  # Now do something with the image! For example, let's display it:
  st.image(opencv_image, channels="RGB")
  save_uploadedfile(uploaded_file)
  resized = mobilenet_v2_preprocess_input(resized)
  img_reshape = resized[np.newaxis,...]
  
  with open(uploaded_file.name,"wb") as f:
    f.write(uploaded_file.getbuffer())
    
      

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

if st.button("Predict"):
      meso = Meso4()
      meso.load("https://github.com/filxy/test/blob/main/saved_model/Meso4_DF")
      
                                      #image generator to predict images
      dataGenerator = ImageDataGenerator(rescale=1./255)
      generator = dataGenerator.flow_from_directory('https://github.com/filxy/test/tree/main/tempDir',
                                                      target_size=(256, 256),
                                                      batch_size= 1,
                                                      class_mode='binary',
                                                      subset='training')
            # Evaluating Prediction
      X, y = generator.next()

                                      
      st.write('Predicted :', meso.predict(X), '\nReal class :', y)
                                          
      st.write("predictions closer to 0 are fake and predictions closer to 1 are real")
