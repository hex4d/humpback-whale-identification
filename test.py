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

base_dir='data'
train_dir=os.path.join(base_dir, 'train')
validation_dir=os.path.join(base_dir, 'validation')
model_base_dir = 'models/'

# paramater
batch_size=20
SEED=1470
epochs=5

input_size = 224
def get_model():
    main_input = layers.Input(shape=(input_size, input_size, 3,))
    conv_base = resnet50.ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(input_size, input_size, 3))(main_input)
    features = layers.GlobalAveragePooling2D()(conv_base)
    features = layers.BatchNormalization()(features)
    h1 = layers.Dense(2048, activation='relu')(features)
    # new or known
    h2_is_new = layers.Dense(512, activation='relu')(h1)
    is_new_output = layers.Dense(1, activation='sigmoid', name='is_new_output')(h2_is_new)
    # classify
    h2_classify = layers.Dense(2048, activation='relu')(h1)
    muled = layers.Concatenate()([is_new_output, h2_classify])
    output = layers.Dense(5004, activation='softmax', name='output')(muled)
    model = models.Model(inputs=[main_input], outputs=[is_new_output, output])
    model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  loss={'is_new_output': 'binary_crossentropy', 'output' : 'categorical_crossentropy'},
                        metrics=['accuracy'])
    return model

def load_data():
    datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
    )
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        seed=SEED,
        class_mode='categorical',
    )
    datagen = ImageDataGenerator(
        rescale=1./255,
    )
    val_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(input_size, input_size),
        batch_size=20,
        seed=SEED,
        class_mode='categorical',
    )
    return train_generator, val_generator

def batch_generator(generator):
    for x, y in generator:
        yield x, [y[:,0], y[:,1:]]


train_generator, val_generator = load_data()
model = get_model()

model_dir = os.path.join(model_base_dir, 'separate_dnn')
os.makedirs(model_dir, exist_ok=True)

model_checkpoint_path = os.path.join(model_dir, '{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5')
model_checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True)
csv_log_path = os.path.join(model_dir, 'train.log')
csv_logger = CSVLogger(csv_log_path)
history = model.fit_generator(batch_generator(train_generator),
                    steps_per_epoch=5,
                    epochs=epochs,
                    validation_data=batch_generator(val_generator),
                    validation_steps=50,
                    callbacks=[model_checkpoint, csv_logger]
                    )
