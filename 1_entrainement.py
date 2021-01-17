#!/usr/bin/env python
# coding: utf-8

# ## Import 

# In[1]:


import tensorflow as tf
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, AveragePooling2D, Activation, Dropout, Flatten, Dense

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
import numpy as np


import mlflow
import mlflow.tensorflow


# ## Telechargement du ZIP

# In[2]:


from modules.loader import Loader

loader = Loader(
    "https://stdatalake010.blob.core.windows.net/public/cifar-100.zip",
    '../datas/ZIP/',
    extraction_target='../datas/RAW/'
)
loader.ensure_data_loaded()


# ## Extraction d'un jeu de donnees

# In[3]:


from modules.splitting import Splitting

labels_array = ['apple', 'bee']

TRAIN_DATA_DIR = Splitting.copie_dossiers(
    '../datas/RAW/train',
    labels_array,
    500,
    explorer=False
)

print(TRAIN_DATA_DIR)


# ## Chargement des images

# In[4]:


image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)

# Taille d'image
TRAIN_IMAGE_SIZE = 32
# NB d'images envoyées à la fois
TRAIN_BATCH_SIZE = 64

train_generator = image_data_generator.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE),
    batch_size=TRAIN_BATCH_SIZE,
    class_mode='categorical',
    subset='training')
 
validation_generator = image_data_generator.flow_from_directory(
    TRAIN_DATA_DIR, # same directory as training data
    target_size=(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE),
    batch_size=TRAIN_BATCH_SIZE,
    class_mode='categorical',
    subset='validation')


# ## Creation du modele
# 
# - convolution
# - dense
# - pooling

# In[5]:


model = Sequential()

model.add(Conv2D(32, kernel_size=3, activation='elu', kernel_initializer='he_uniform', padding='same', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(64, activation='elu'))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=3, activation='elu', kernel_initializer='he_uniform', padding='same'))
model.add(Dense(128, activation='elu'))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=3, activation='elu', kernel_initializer='he_uniform', padding='same'))

#Toujours à la fin
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

# model.summary()


# In[6]:


from tensorflow.keras.callbacks import EarlyStopping

with mlflow.start_run():

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    es_callback = EarlyStopping(monitor='val_loss', patience=15)
    training = model.fit(train_generator, epochs=40, callbacks=[es_callback], validation_data=validation_generator, shuffle=False)
    mlflow.tensorflow.autolog()


# In[7]:


# plt.plot(training.history['accuracy'], color='red', label='Training accuracy')
# plt.plot(training.history['val_accuracy'],  color='green', label='Validation accuracy')

# plt.xlabel('Epochs')
# plt.ylabel('Loss')

# plt.legend()
# plt.ylim((0,1.01))

# plt.show()


# In[8]:


# plt.plot(training.history['loss'], color='red', label='Training loss')
# plt.plot(training.history['val_loss'],  color='green', label='Validation loss')

# plt.xlabel('Epochs')
# plt.ylabel('Loss')

# plt.legend()

# plt.show()


# ## Sauvegarde du modele

# In[9]:


model_name = 'model_apple_bee.h5'

model.save(model_name)

# model.summary()


# ## Affichage des couches du modele

# In[10]:


# from modules.observation_modele import plot_layer

# img = "../datas/RAW/train/apple/0020.png"
# plt.imshow(plt.imread(img))

# plot_layer(model, img, range(8))


# In[11]:


# from modules.observation_modele import plot_layer

# img = "../datas/RAW/train/bee/0103.png"
# plt.imshow(plt.imread(img))

# plot_layer(model, img, range(8))

