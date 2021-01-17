#!/usr/bin/env python
# coding: utf-8

# # MLflow Training Tutorial (la copie)

# In[1]:


def train(labels_array, nb_epochs, nb_patience):

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, AveragePooling2D, Activation, Dropout, Flatten, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.preprocessing import image

    import mlflow
    import mlflow.tensorflow

    # Telechargement du ZIP
    from modules.loader import Loader

    loader = Loader(
        "https://stdatalake010.blob.core.windows.net/public/cifar-100.zip",
        '../datas/ZIP/',
        extraction_target='../datas/RAW/'
    )
    loader.ensure_data_loaded()

    # Extraction du jeu de donnees
    from modules.splitting import Splitting

    labels_array = ['apple', 'bee']

    TRAIN_DATA_DIR = Splitting.copie_dossiers(
        '../datas/RAW/train',
        labels_array,
        500,
        explorer=False
    )

    print(TRAIN_DATA_DIR)

    # Chargement des images
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)
    TRAIN_IMAGE_SIZE = 32
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

    with mlflow.start_run():

        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, activation='elu', kernel_initializer='he_uniform', padding='same', input_shape=(32,32,3)))
        #Toujours à la fin
        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        es_callback = EarlyStopping(monitor='val_loss', patience=nb_patience)
        training = model.fit(train_generator, epochs=nb_epochs, callbacks=[es_callback], validation_data=validation_generator, shuffle=False)
        
        # mlflow.log_param("labels_array", labels_array)
        # mlflow.log_param("nb_epochs", nb_epochs)
        # mlflow.log_param("nb_patience", nb_patience)

        # mlflow.log_metric("accuracy", training.history['accuracy'])
        # mlflow.log_metric("val_accuracy", training.history['val_accuracy'])
        # mlflow.log_metric("loss", training.history['loss'])
        # mlflow.log_metric("val_loss", training.history['val_loss'])


import sys

if len(sys.argv) != 3:
    raise Exception("Args Number is invalid !")

labels_array = sys.argv[0]
nb_epochs = sys.argv[1]
nb_patience = sys.argv[2]

import mlflow
mlflow.tensorflow.autolog()

train(labels_array, nb_epochs, nb_patience)

