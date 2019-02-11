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

import argparse

# settings
base_dir='../../data'
model_base_dir = 'models'
test_dir = os.path.join(base_dir, 'test')
train_dir=os.path.join(base_dir, 'train')
validation_dir=os.path.join(base_dir, 'validation')
# paramater
batch_size=32
SEED=1470
epochs=50
input_size = 224

def split_with_class_count(df, validation_split=0.05):
    df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    df = df[df.Id != 'new_whale']
    classes = df['Id'].unique()
    df['count'] = df.groupby('Id')['Id'].transform('count')
    fdf = df[df['count'] >= 2]
    val_classes = fdf['Id'].unique()
    train_df = pd.DataFrame(columns=df.columns)
    train_df = pd.concat([train_df, df])
    validation_df = pd.DataFrame(columns=df.columns)
    for val_class in val_classes:
      class_df = df[df.Id == val_class]
      validation = class_df.sample(frac=validation_split, random_state=SEED)
      validation_df = pd.concat([validation_df, validation]) 
      train_df = train_df.drop(validation.index)
    train_df = train_df.drop('count', axis=1)
    train_df = train_df.reset_index()
    validation_df = validation_df.drop('count', axis=1)
    validation_df = validation_df.reset_index()
    print('train', len(train_df), 'validation', len(validation_df))
    return train_df, validation_df, classes.tolist()

def load_data():
    df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    df = df[df.Id != 'new_whale'] # without new_whale
    train_df, validation_df, classes = split_with_class_count(df, validation_split=0.1, class_count=50)
    print(train_df)
    print(df)
    datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        brightness_range=[0.7, 1.0],
    )
    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=train_dir,
        x_col='Image',
        y_col='Id',
        target_size=(input_size, input_size),
        batch_size=batch_size,
        classes=classes,
        seed=SEED,
    )
    datagen = ImageDataGenerator(
        rescale=1./255,
    )
    val_generator = datagen.flow_from_dataframe(
        dataframe=validation_df,
        directory=train_dir,
        x_col='Image',
        y_col='Id', 
        target_size=(input_size, input_size),
        batch_size=20,
        classes=classes,
        seed=SEED,
    )
    return train_generator, val_generator

from keras import regularizers
class ModelV7():
    def __init__(self):
        self.name = 'v7'
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
        return model

from keras import regularizers
class ModelV12():
    def __init__(self):
        self.name = 'v12'
    def init_model(self):
        model = models.load_model('./pretrained.hdf5')
        pretrained_weights = model.get_weights()
        model = ModelV7().init_model()
        model.set_weights(pretrained_weights)
        trimmed_model = models.Model(inputs=model.input, outputs=model.layers[-4].output)
        trimmed_model.get_layer(name='dropout_1').name = 'pre_dropout1'
        trimmed_model.get_layer(name='dense_1').name = 'pre_dense1'
        for layer in trimmed_model.layers:
            layer.trainable = False
        for layer in trimmed_model.layers[-6:]:
            layer.trainable = True
            print(layer)
        main_input = trimmed_model.input
        embedding = trimmed_model.output
        x = layers.BatchNormalization()(embedding)
        x = layers.Dense(2024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(5005, activation='softmax')(x)
        model = models.Model(inputs=[main_input], outputs=[x])
        model.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        self.model = model
        return model
    def train(self, train_generator, val_generator, callbacks, epochs=100, initial_epoch=None):
        epochs1 = int(epochs * 0.9)
        if initial_epoch is None:
            initial_epochs = 0
        if initial_epoch < epochs1:
            inicial_epoch2 = epochs1
        else:
            initial_epoch2 = initial_epoch
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs1,
            validation_data=val_generator,
            validation_steps=50,
            callbacks=callbacks,
            initial_epoch=initial_epoch
        )
        for layer in self.model.layers[-17:]:
            layer.trainable=True
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['acc'])
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=50,
            callbacks=callbacks,
            initial_epoch=initial_epoch,
        )


def load_classes():
    train_generator, val_generator = load_data()
    return np.array([c for c, v in train_generator.class_indices.items()])

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
        seed=SEED,
    )
    if len(test_generator) == 0:
        print('Train data not found')
        exit()
    return test_generator

def predict(model, generator):
    kth = 5
    generator.reset()
    pred = model.predict_generator(generator, steps=len(generator), verbose=1)
    classes = load_classes()
    classify_index = np.argpartition(-pred, kth)[:, :kth]
    classify_value = pred[np.arange(pred.shape[0])[:, None], classify_index]
    best_5_pred = np.zeros((len(classify_index), 5))
    best_5_class = np.zeros((len(classify_index), 5), dtype='int32')
    for i, p in enumerate(classify_value):
        sort_index = np.argsort(p)[::-1]
        best_5_pred[i] = (p[sort_index])
        best_5_class[i] = (classify_index[i][sort_index])
    ### kokokoko?
    # create output
    submit = pd.DataFrame(columns=['Image', 'Id'])
    for i, p in enumerate(best_5_pred):
        submit_classes = []
        if p[0] < 0.55:
            submit_classes.append('new_whale')
            submit_classes.extend(classes[best_5_class[i]][0:4])
        elif p[1] < 0.4 :
            submit_classes.extend(classes[best_5_class[i]][0:1])
            submit_classes.append('new_whale')
            submit_classes.extend(classes[best_5_class[i]][1:4])
        elif p[2] < 0.1 :
            submit_classes.extend(classes[best_5_class[i]][0:2])
            submit_classes.append('new_whale')
            submit_classes.extend(classes[best_5_class[i]][2:4])
        elif p[3] < 0.05 :
            submit_classes.extend(classes[best_5_class[i]][0:3])
            submit_classes.append('new_whale')
            submit_classes.extend(classes[best_5_class[i]][3:4])
        else:
            submit_classes.extend(classes[best_5_class[i]])
        classes_text = ' '.join(submit_classes)
        submit = submit.append(pd.Series(np.array([generator.filenames[i].split('/')[1], classes_text]), index=submit.columns), ignore_index=True)
    return submit

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', help='optional', action='store_true')
    args = parser.parse_args()
    print(args)
    if args.pred:
        # load models
        # model = models.load_model('./model_dir/.hdf5')
        # load test data
        test_generator = load_test_data()
        # predict
        submit = predict(model, test_generator)
        submit.to_csv('submit4.csv', index=False)
    else:
        train_generator, val_generator = load_data()
        model_wrapper = ModelV10_1()
        model = model_wrapper.init_model()
        model.summary()

        model_dir = os.path.join(model_base_dir, 'with_whale' + model_wrapper.name)

        os.makedirs(model_dir, exist_ok=True)

        model_checkpoint_path = os.path.join(model_dir, '{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5')
        model_checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True)
        tensor_board = TensorBoard(log_dir=os.path.join(model_dir))
        model_wrapper.train(train_generator, val_generator, [model_checkpoint, tensor_board], epochs=100, initial_epoch=0)
        model.save(os.path.join(model_dir, str(100) + 'model.hdf5'))
