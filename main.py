from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
import os
import matplotlib.pyplot as plt
import pandas as pd

base_dir='data'
train_dir=os.path.join(base_dir, 'train')
validation_dir=os.path.join(base_dir, 'validation')

# paramater
input_size=225
batch_size=20
SEED=1470
epochs=5
validation_split=0.1

def load_data():
    datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=10,
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
        batch_size=2,
        seed=SEED,
        class_mode='categorical',
    )
    return train_generator, val_generator

def vgg_cnn_model(input_size, output_size):
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(input_size, input_size, 3))
    conv_base.trainable = False
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(output_size))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    return model

def show_image(image):
    plt.imshow(image)
    plt.show()

train_generator, val_generator = load_data()
classes= train_generator.class_indices
df = pd.DataFrame.from_dict(classes, orient='index')
df.to_csv('train_classes.cls')

model = vgg_cnn_model(input_size, len(classes))

# model.fit_generator(train_generator, steps_per_epoch=2, epochs=epochs, validation_data=val_generator, validation_steps=50)
model.fit_generator(train_generator, steps_per_epoch=2, epochs=epochs)

model_path = 'models/'
if not os.path.exists(model_path):
    os.mkdir(model_path)
model_path = os.path.join(model_path, 'model.h5')
model.save(model_path)

