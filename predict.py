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

test_dir = 'data/test/'
input_size = 224
seed = 1470

def load_test_data():
    datagen = ImageDataGenerator(
        rescale=1./255,
    )
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(input_size, input_size),
        batch_size=1,
        class_mode=None,
        shuffle=False,
        seed=seed,
    )
    return test_generator

new_known_model = models.load_model('models/separate1_new_known/model.h5')
classify_model = models.load_model('models/separate2_classify/115-4.96-4.36.hdf5')

new_known_model.summary()
classify_model.summary()

test_generator = load_test_data()

test_generator.reset()
pred = new_known_model.predict_generator(test_generator, steps=len(test_generator),verbose=1)
pred = classify_model(test_generator, verbose=1)
