from keras.preprocessing.image import ImageDataGenerator
from keras.applications import resnet50
from keras import models
from keras import layers
from keras import optimizers
from keras.engine.topology import Input
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# settings
base_dir='../../data'
model_base_dir = 'models'
# paramater
batch_size=32
SEED=1470
epochs=50
input_size = 224

train_dir=os.path.join(base_dir, 'train')
validation_dir=os.path.join(base_dir, 'validation')

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

from keras import regularizers
class ModelV8():
    def __init__(self):
        self.name = 'v8'
    def init_model(self):
        conv_base = resnet50.ResNet50(weights='imagenet',
                                      include_top=False,
                                      input_shape=(input_size, input_size, 3))
        for layer in conv_base.layers[:]:
            if 'BatchNormalization' in str(layer):
                layer.trainable = True
            else:
                layer.trainable = False
        main_input = conv_base.input
        embedding = conv_base.output
        x = layers.GlobalMaxPooling2D()(embedding)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(5004, activation='softmax')(x)
        model = models.Model(inputs=[main_input], outputs=[x])
        model.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        self.model = model
        return model
    def train(self, train_generator, val_generator, callbacks):
        epochs = 50
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=10,
            validation_data=val_generator,
            validation_steps=50,
            callbacks=callbacks,
        )
        for layer in self.model.layers[-17:]:
            layer.trainable=True
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['acc'])
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=40,
            validation_data=val_generator,
            validation_steps=50,
            callbacks=callbacks,
            initial_epoch=10
        )

if __name__ == '__main__':
    train_generator, val_generator = load_data()
    model_wrapper = ModelV8()
    model_wrapper.init_model()
    model_wrapper.model.summary()

    model_dir = os.path.join(model_base_dir, 'without_whale' + model_wrapper.name)

    os.makedirs(model_dir, exist_ok=True)

    model_checkpoint_path = os.path.join(model_dir, 'model.hdf5')
    model_checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True)
    tensor_board = TensorBoard(log_dir=os.path.join(model_dir))
    model_wrapper.train(train_generator, val_generator, [model_checkpoint, tensor_board])

