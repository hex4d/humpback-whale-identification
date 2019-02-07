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
model_path = 'models/'

# paramater
input_size=224
batch_size=4
SEED=1470
epochs=50

def load_data():
    datagen = ImageDataGenerator(
       rescale=1./255,
        horizontal_flip=True,
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

def resnet_cnn_model(input_size, output_size):
    conv_base = resnet50.ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(input_size, input_size, 3))
    for layer in conv_base.layers[:139]: # default 179
       if 'BatchNormalization' in str(layer):
          layer.trainable = True
       else:
         layer.trainable = False
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(output_size))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['acc'])
    return model

def show_image(image):
    plt.imshow(image)
    plt.show()

def mount_drive():
    from google.colab import drive
    drive.mount('/content/drive')


COLABO = False

if COLABO:
    mount_drive()

train_generator, val_generator = load_data()
classes= train_generator.class_indices
df = pd.DataFrame.from_dict(classes, orient='index')
df.to_csv('train_classes.cls')

model = resnet_cnn_model(input_size, len(classes))

# model.fit_generator(train_generator, steps_per_epoch=len(train_generator), epochs=epochs, validation_data=val_generator, validation_steps=50)
if COLABO:
    model_path = os.path.join('drive/My Drive', model_path)
model_path = os.path.join(model_path, 'cnn_model1_2_2')
os.makedirs(model_path, exist_ok=True)
model_checkpoint_path = os.path.join(model_path, '{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5')
model_checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True)
csv_log_path = os.path.join(model_path, 'train.log')
csv_logger = CSVLogger(csv_log_path)

history = model.fit_generator(train_generator, steps_per_epoch=len(train_generator), epochs=epochs, validation_data=val_generator, validation_steps=50, callbacks=[model_checkpoint, csv_logger])

if COLABO:
    shutil.move(model_path, 'drive/My Drive')

