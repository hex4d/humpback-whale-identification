from keras.preprocessing.image import ImageDataGenerator
from keras.applications import resnet50
from keras import models
from keras import layers
from keras import optimizers
from keras.engine.topology import Input
from keras.callbacks import ModelCheckpoint, CSVLogger

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DATA_BASE_DIR='data'
TRAIN_DIR=os.path.join(DATA_BASE_DIR, 'train')
VALIDATION_DIR=os.path.join(DATA_BASE_DIR, 'validation')
MODEL_BASE_DIR = 'models/'

# paramater
BATCH_SIZE=32
SEED=1470
EPOCHS=50
MODEL_ID = 0
INPUT_SIZE = 224

def get_model():
    conv_base = resnet50.ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
    for layer in conv_base.layers[:]:
        if 'BatchNormalization' in str(layer):
            layer.trainable = True
        else:
            layer.trainable = False
    main_input = conv_base.input
    embedding = conv_base.output
    x = layers.GlobalAveragePooling2D()(embedding)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=[main_input], outputs=[x])
    model.compile(
        loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def load_data():
    datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        brightness_range=[0.7, 1.0],
    )
    train_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(INPUT_SIZE, INPUT_SIZE),
        batch_size=BATCH_SIZE,
        seed=SEED,
        class_mode='categorical',
    )
    datagen = ImageDataGenerator(
        rescale=1./255,
    )
    val_generator = datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(INPUT_SIZE, INPUT_SIZE),
        batch_size=20,
        seed=SEED,
        class_mode='categorical',
    )
    return train_generator, val_generator

def batch_generator(generator):
    for x, y in generator:
        yield x, [y[:,0], y[:,0:]]


train_generator, val_generator = load_data()
model = get_model()

model_dir = os.path.join(MODEL_BASE_DIR, 'new_whale' + str(MODEL_ID))

os.makedirs(model_dir, exist_ok=True)

model_checkpoint_path = os.path.join(model_dir, '{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5')
model_checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True)
csv_log_path = os.path.join(model_dir, 'train.log')
csv_logger = CSVLogger(csv_log_path)
history = model.fit_generator(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=EPOCHS,
                    validation_data=val_generator,
                    validation_steps=50,
                    callbacks=[model_checkpoint, csv_logger]
                    )

