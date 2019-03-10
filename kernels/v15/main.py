from keras.preprocessing.image import ImageDataGenerator
from keras.applications import resnet50
from keras import models
from keras import layers
from keras import optimizers
from keras.engine.topology import Input
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, ReduceLROnPlateau
from keras import backend as K

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import argparse

# settings
base_dir='../../data'
model_base_dir = 'models'
test_dir = os.path.join(base_dir, 'test_cropped/0')
train_dir=os.path.join(base_dir, 'cropped')
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
    train_df, validation_df, classes = split_with_class_count(df, validation_split=0.1)
    datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        brightness_range=[0.7, 1.0],
    )
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
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
class ModelV15():
    def __init__(self):
        self.name = 'v15'
    def get_model(self):
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

def load_classes():
    train_generator, val_generator = load_data()
    return np.array([c for c, v in train_generator.class_indices.items()])

def load_test_data():
    datagen = ImageDataGenerator(
        rescale=1./255,
    )
    pd.DataFrame(os.listdir(test_dir),columns=['filename'])
    test_generator = datagen.flow_from_dataframe(
        pd.DataFrame(os.listdir(test_dir),columns=['filename']),
        test_dir,
        target_size=(input_size, input_size),
        batch_size=32,
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
    pred = model.predict_generator(test_batch_generator(generator), steps=len(generator), verbose=1)
    print(pred[0:10])
    classes = load_classes()
    classify_index = np.argpartition(-pred, kth)[:, :kth]
    classify_value = pred[np.arange(pred.shape[0])[:, None], classify_index]
    print(classify_value[0:10], classify_index[0:10])
    best_5_pred = np.zeros((len(classify_index), 5))
    best_5_class = np.zeros((len(classify_index), 5), dtype='int32')
    for i, p in enumerate(classify_value):
        sort_index = np.argsort(p)[::-1]
        best_5_pred[i] = (p[sort_index])
        best_5_class[i] = (classify_index[i][sort_index])
    # create output
    print(best_5_class[0:10])
    submit = pd.DataFrame(columns=['Image', 'Id'])
    for i, p in enumerate(best_5_pred):
        submit_classes = []
        if p[0] < 0.55:
            submit_classes.append('new_whale')
            submit_classes.extend(classes[best_5_class[i]][0:4])
        elif p[1] < 0.35 :
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
        submit = submit.append(pd.Series(np.array([generator.filenames[i], classes_text]), index=submit.columns), ignore_index=True)
    return submit

def test_batch_generator(generator):
    for x in generator:
        print(x)
        x = (x - np.average(x, axis=0)) / (np.std(x, axis=0) + K.epsilon())
        yield x

def batch_generator(generator):
    for x, y in generator:
        x = (x - np.average(x, axis=0)) / np.std(x, axis=0)
        yield x, y


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', help='optional', action='store_true')
    args = parser.parse_args()
    print(args)
    if args.pred:
        # load models
        model = models.load_model('./models/iterate_without_whalev15/144-2.78-3.22.hdf5')
        # load test data
        generator = load_test_data()
        # predict
        submit = predict(model, generator)
        submit.to_csv('submit.csv', index=False)
    else:
        epochs = 50
        train_generator, val_generator = load_data()
        model_wrapper = ModelV15()
        model = model_wrapper.get_model()
        model.summary()

        model_dir = os.path.join(model_base_dir, 'iterate_without_whale' + model_wrapper.name)
        os.makedirs(model_dir, exist_ok=True)
        for num in range(1, 4):
            model_checkpoint_path = os.path.join(model_dir, str(num) + '{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5' )
            model_checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True)
            tensor_board = TensorBoard(log_dir=os.path.join(model_dir))
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, min_delta=0.1, factor=0.2, min_lr=0.0005, verbose=1)
            history = model.fit_generator(
                batch_generator(train_generator),
                steps_per_epoch=len(train_generator),
                epochs=epochs,
                validation_data=batch_generator(val_generator),
                validation_steps=50,
                callbacks=[model_checkpoint, tensor_board, reduce_lr],
                initial_epoch=(num-1)*epochs
            )
