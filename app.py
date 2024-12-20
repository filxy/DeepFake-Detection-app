import streamlit as st
import sklearn
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
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from scipy.ndimage.interpolation import zoom,rotate
import random
import os
from os import listdir
from os.path import isfile, join
from pytube import YouTube

    

from math import floor
from scipy.ndimage.interpolation import zoom, rotate
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
  with open(os.path.join("test",uploaded_file.name),"wb") as f:
    f.write(uploaded_file.getbuffer())
    return st.success("Saved File:{} to test".format(uploaded_file.name))

def main():
  Menu = ["Image Prediction","Web-cam prediction","Youtube Prediction","About"]
  choice = st.sidebar.selectbox("Menu",Menu)

  if choice == "Image Prediction":
    st.subheader("Predict deep fake images")
    uploaded_file = st.file_uploader("Choose a image file", type = ['jpeg','jpg','PNG'])
    if uploaded_file is not None:
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
                                      
          x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
                                                    
          x1 = BatchNormalization()(x1)
                                                    
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
        meso.load("saved_model/Meso4_DF")
        
        
        dataGenerator = ImageDataGenerator(rescale=1./255)
        generator = dataGenerator.flow_from_directory('test',
                                                      target_size=(256, 256),
                                                      batch_size= 1,
                                                      class_mode='binary',
                                                      subset='training')
        # Evaluating Prediction
        X, y = generator.next()

                                            
        st.write('Predicted :', meso.predict(X), '\nReal class :', y)
                                                
        st.write("predictions closer to 0 are fake and predictions closer to 1 are real") 
                    
  

  elif choice == "Web-cam prediction":
    st.subheader("Predict with web-cam")
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
      
      bytes_data = img_file_buffer.getvalue()
      cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
      def js_to_image(js_reply):
        image_bytes = b64decode(js_reply.split(',')[1])
        jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=1)
        return img
      def bbox_to_bytes(bbox_array):
        bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
        iobuf = io.BytesIO()
        bbox_PIL.save(iobuf, format='png')
        bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))
        return bbox_bytes

      face_cascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
      def take_photo(filename='photo.jpg', quality=0.8):
        js = Javascript('''
          async function takePhoto(quality) {
            const div = document.createElement('div');
            const capture = document.createElement('button');
            capture.textContent = 'Capture';
            div.appendChild(capture);

            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});

            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            // Resize the output to fit the video element.
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

            // Wait for Capture to be clicked.
            await new Promise((resolve) => capture.onclick = resolve);

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getVideoTracks()[0].stop();
            div.remove();
            return canvas.toDataURL('image/jpeg', quality);
          }
          ''')
        display(js)
        
        data = eval_js('takePhoto({})'.format(quality))
  
        img = js_to_image(data) 
  
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        st.write(gray.shape)
  
        faces = face_cascade.detectMultiScale(gray)
  
        for (x,y,w,h) in faces:
          img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
          cv2.imwrite(filename, img)
          return filename

        try:
          filename = take_photo('test/Photo.jpg')
          st.write('Saved to {}'.format(filename))
          display(Image(filename))
        except Exception as err:
          st.write(str(err))

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
      meso.load("saved_model/Meso4_DF")
      
      
      dataGenerator = ImageDataGenerator(rescale=1./255)
      generator = dataGenerator.flow_from_directory('test',
                                                    target_size=(256, 256),
                                                    batch_size= 1,
                                                    class_mode='binary',
                                                    subset='training')
                # Evaluating Prediction
      X, y = generator.next()

                                          
      st.write('Predicted :', meso.predict(X), '\nReal class :', y)
                                              
      st.write("predictions closer to 0 are fake and predictions closer to 1 are real")

  elif choice == "Youtube Prediction":
    st.subheader("Predict with Youtube Url")
    st.subheader("WARNING - youtube prediction is currently unavailable due to technical difficulties")
    videourl = st.text_input('Type youtube url here',"https://www.youtube.com/watch?v=DdZ163jzw4w")
    st.video(videourl)

    def downloadYouTube(videourl, path):
      yt = YouTube(videourl)
      yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
      if not os.path.exists(path):
        os.makedirs(path)
      yt.download(path)
      downloadYouTube(videourl, 'tempDir')

    if st.button("Predict Youtube"):
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

      ## Face extraction
      class Video:
        def __init__(self, path):
          self.path = path
          self.container = imageio.get_reader(path, 'ffmpeg')
          self.length = self.container.count_frames()
          self.fps = self.container.get_meta_data()['fps']
            
        def init_head(self):
          self.container.set_image_index(0)
            
        def next_frame(self):
          self.container.get_next_data()
            
        def get(self, key):
          return self.container.get_data(key)
            
        def __call__(self, key):
          return self.get(key)
            
        def __len__(self):
          return self.length


      class FaceFinder(Video):
        def __init__(self, path, load_first_face = True):
          super().__init__(path)
          self.faces = {}
          self.coordinates = {}  
          self.last_frame = self.get(0)
          self.frame_shape = self.last_frame.shape[:2]
          self.last_location = (0, 200, 200, 0)
          if (load_first_face):
            face_positions = face_recognition.face_locations(self.last_frame, number_of_times_to_upsample=2)
            if len(face_positions) > 0:
              self.last_location = face_positions[0]
            
        def load_coordinates(self, filename):
          np_coords = np.load(filename)
          self.coordinates = np_coords.item()
            
        def expand_location_zone(self, loc, margin = 0.2):
          ''' Adds a margin around a frame slice '''
          offset = round(margin * (loc[2] - loc[0]))
          y0 = max(loc[0] - offset, 0)
          x1 = min(loc[1] + offset, self.frame_shape[1])
          y1 = min(loc[2] + offset, self.frame_shape[0])
          x0 = max(loc[3] - offset, 0)
          return (y0, x1, y1, x0)
            
        @staticmethod
        def upsample_location(reduced_location, upsampled_origin, factor):
          ''' Adapt a location to an upsampled image slice '''
          y0, x1, y1, x0 = reduced_location
          Y0 = round(upsampled_origin[0] + y0 * factor)
          X1 = round(upsampled_origin[1] + x1 * factor)
          Y1 = round(upsampled_origin[0] + y1 * factor)
          X0 = round(upsampled_origin[1] + x0 * factor)
          return (Y0, X1, Y1, X0)

        @staticmethod
        def pop_largest_location(location_list):
          max_location = location_list[0]
          max_size = 0
          if len(location_list) > 1:
            for location in location_list:
              size = location[2] - location[0]
              if size > max_size:
                max_size = size
                max_location = location
          return max_location
            
        @staticmethod
        def L2(A, B):
          return np.sqrt(np.sum(np.square(A - B)))
            
        def find_coordinates(self, landmark, K = 2.2):
          '''
          We either choose K * distance(eyes, mouth),
          or, if the head is tilted, K * distance(eye 1, eye 2)
          /!\ landmarks coordinates are in (x,y) not (y,x)
          '''
          E1 = np.mean(landmark['left_eye'], axis=0)
          E2 = np.mean(landmark['right_eye'], axis=0)
          E = (E1 + E2) / 2
          N = np.mean(landmark['nose_tip'], axis=0) / 2 + np.mean(landmark['nose_bridge'], axis=0) / 2
          B1 = np.mean(landmark['top_lip'], axis=0)
          B2 = np.mean(landmark['bottom_lip'], axis=0)
          B = (B1 + B2) / 2

          C = N
          l1 = self.L2(E1, E2)
          l2 = self.L2(B, E)
          l = max(l1, l2) * K
          if (B[1] == E[1]):
            if (B[0] > E[0]):
              rot = 90
            else:
              rot = -90
          else:
            rot = np.arctan((B[0] - E[0]) / (B[1] - E[1])) / np.pi * 180
                
          return ((floor(C[1]), floor(C[0])), floor(l), rot)
            
        def find_faces(self, resize = 0.5, stop = 0, skipstep = 0, no_face_acceleration_threshold = 3, cut_left = 0, cut_right = -1, use_frameset = False, frameset = []):
          '''
          The core function to extract faces from frames
          using previous frame location and downsampling to accelerate the loop.
          '''
          not_found = 0
          no_face = 0
          no_face_acc = 0
                
          # to only deal with a subset of a video, for instance I-frames only
          if (use_frameset):
            finder_frameset = frameset
          else:
            if (stop != 0):
              finder_frameset = range(0, min(self.length, stop), skipstep + 1)
            else:
              finder_frameset = range(0, self.length, skipstep + 1)
          # Quick face finder loop
          for i in finder_frameset:
            # Get frame
            frame = self.get(i)
            if (cut_left != 0 or cut_right != -1):
              frame[:, :cut_left] = 0
              frame[:, cut_right:] = 0            
                    
            # Find face in the previously found zone
            potential_location = self.expand_location_zone(self.last_location)
            potential_face_patch = frame[potential_location[0]:potential_location[2], potential_location[3]:potential_location[1]]
            potential_face_patch_origin = (potential_location[0], potential_location[3])
            
            reduced_potential_face_patch = zoom(potential_face_patch, (resize, resize, 1))
            reduced_face_locations = face_recognition.face_locations(reduced_potential_face_patch, model = 'cnn')
                    
            if len(reduced_face_locations) > 0:
              no_face_acc = 0  # reset the no_face_acceleration mode accumulator

              reduced_face_location = self.pop_largest_location(reduced_face_locations)
              face_location = self.upsample_location(reduced_face_location,
                                                     potential_face_patch_origin,
                                                     1 / resize)
              self.faces[i] = face_location
              self.last_location = face_location
                        
              # extract face rotation, length and center from landmarks
              landmarks = face_recognition.face_landmarks(frame, [face_location])
              if len(landmarks) > 0:
                # we assume that there is one and only one landmark group
                self.coordinates[i] = self.find_coordinates(landmarks[0])
            else:
              not_found += 1

              if no_face_acc < no_face_acceleration_threshold:
                # Look for face in full frame
                face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample = 2)
              else:
                # Avoid spending to much time on a long scene without faces
                reduced_frame = zoom(frame, (resize, resize, 1))
                face_locations = face_recognition.face_locations(reduced_frame)
                            
              if len(face_locations) > 0:
                st.write('Face extraction warning : ', i, '- found face in full frame', face_locations)
                no_face_acc = 0  
                            
                face_location = self.pop_largest_location(face_locations)
                            
                # if was found on a reduced frame, upsample location
                if no_face_acc > no_face_acceleration_threshold:
                  face_location = self.upsample_location(face_location, (0, 0), 1 / resize)
                            
                  self.faces[i] = face_location
                  self.last_location = face_location
                            
                  # extract face rotation, length and center from landmarks
                  landmarks = face_recognition.face_landmarks(frame, [face_location])
                  if len(landmarks) > 0:
                    self.coordinates[i] = self.find_coordinates(landmarks[0])
              else:
                st.write('Face extraction warning : ',i, '- no face')
                no_face_acc += 1
                no_face += 1

          st.write('Face extraction report of', 'not_found :', not_found)
          st.write('Face extraction report of', 'no_face :', no_face)
          return 0
            
        def get_face(self, i):
          ''' Basic unused face extraction without alignment '''
          frame = self.get(i)
          if i in self.faces:
            loc = self.faces[i]
            patch = frame[loc[0]:loc[2], loc[3]:loc[1]]
            return patch
          return frame
            
        @staticmethod
        def get_image_slice(img, y0, y1, x0, x1):
          '''Get values outside the domain of an image'''
          m, n = img.shape[:2]
          padding = max(-y0, y1-m, -x0, x1-n, 0)
          padded_img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
          return padded_img[(padding + y0):(padding + y1),
                            (padding + x0):(padding + x1)]
            
        def get_aligned_face(self, i, l_factor = 1.3):
          '''
          The second core function that converts the data from self.coordinates into an face image.
          '''
          frame = self.get(i)
          if i in self.coordinates:
            c, l, r = self.coordinates[i]
            l = int(l) * l_factor # fine-tuning the face zoom we really want
            dl_ = floor(np.sqrt(2) * l / 2) # largest zone even when rotated
            patch = self.get_image_slice(frame,
                                         floor(c[0] - dl_),
                                         floor(c[0] + dl_),
                                         floor(c[1] - dl_),
                                         floor(c[1] + dl_))
            rotated_patch = rotate(patch, -r, reshape=False)
            # note : dl_ is the center of the patch of length 2dl_
            return self.get_image_slice(rotated_patch,
                                        floor(dl_-l//2),
                                        floor(dl_+l//2),
                                        floor(dl_-l//2),
                                        floor(dl_+l//2))
          return frame


      ## Face prediction
      class FaceBatchGenerator:
        '''
        Made to deal with framesubsets of video.
        '''
        def __init__(self, face_finder, target_size = 256):
          self.finder = face_finder
          self.target_size = target_size
          self.head = 0
          self.length = int(face_finder.length)

        def resize_patch(self, patch):
          m, n = patch.shape[:2]
          return zoom(patch, (self.target_size / m, self.target_size / n, 1))
            
        def next_batch(self, batch_size = 150):
          batch = np.zeros((1, self.target_size, self.target_size, 3))
          stop = min(self.head + batch_size, self.length)
          i = 0
          while (i < batch_size) and (self.head < self.length):
            if self.head in self.finder.coordinates:
              patch = self.finder.get_aligned_face(self.head)
              batch = np.concatenate((batch, np.expand_dims(self.resize_patch(patch), axis = 0)),
                                     axis = 0)
              i += 1
            self.head += 1
          return batch[1:]

        def predict_faces(generator, Classifier, batch_size = 50, output_size = 1):
          '''
          Compute predictions for a face batch generator
          '''
          n = len(generator.finder.coordinates.items())
          profile = np.zeros((1, output_size))
          for epoch in range(n // batch_size + 1):
            face_batch = generator.next_batch(batch_size = batch_size)
            prediction = Classifier.predict(face_batch)
            if (len(prediction) > 0):
              profile = np.concatenate((profile, prediction))
          return profile[1:]

        def compute_accuracy(Classifier, dirname, frame_subsample_count = 30):
          '''
          Extraction + Prediction over a video
          '''
          filenames = [f for f in listdir(dirname) if isfile(join(dirname, f)) and ((f[-4:] == '.mp4') or (f[-4:] == '.avi') or (f[-4:] == '.mov'))]
          predictions = {}      
          for vid in filenames:
            st.write('Dealing with video ', vid)
            # Compute face locations and store them in the face finder
            face_finder = FaceFinder(join(dirname, vid), load_first_face = False)
            skipstep = max(floor(face_finder.length / frame_subsample_count), 0)
            face_finder.find_faces(resize=0.5, skipstep = skipstep)
            st.write('Predicting ', vid)
            gen = FaceBatchGenerator(face_finder)
            p = predict_faces(gen, Classifier)
        
            predictions[vid[:-4]]=(np.mean(p > 0.5), p)
            #predictions.append((np.mean(p > 0.5), p))
          return predictions
            
           
          Classifier = Meso4()
          Classifier.load("saved_model/Meso4_F2F.h5")
            
          predictions = compute_accuracy(Classifier,'tempDir')
          for video_name in predictions:
            st.write('`{}` video class prediction :'.format(video_name), predictions[video_name][0])
      
  
  elif choice == "About":
    st.write("This app Predicts if an image or video is manipulated or authentic.")
    st.write("It also predicts youtube url and real-time webcam images.")
      
if __name__ == '__main__':
  main()
