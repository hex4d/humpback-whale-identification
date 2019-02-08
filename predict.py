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

def predict(model, generator):
    kth = 5
    generator.reset()
    pred = model.predict_generator(generator, steps=len(generator), verbose=1)
    df = pd.read_csv('train_classes_classify.cls')
    classes = df.loc[:, 'Unnamed: 0']
    classify_index = np.argpartition(-pred, kth)[:, :kth]
    classify_value = pred[np.arange(pred.shape[0])[:, None], classify_index]
    best_5_pred = np.zeros((len(classify_index), 5))
    best_5_class = np.zeros((len(classify_index), 5))
    for i, p in enumerate(classify_value):
        sort_index = np.argsort(p)[::-1]
        best_5_pred[i] = (p[sort_index])
        best_5_class[i] = (classify_index[i][sort_index])
    # create output
    submit = pd.DataFrame(columns=['Image', 'Id'])
    for i, p in enumerate(best_5_pred):
        submit_classes = []
        if p[0] < 0.45:
            submit_classes.append('new_whale')
            submit_classes.extend(classes[classify_index[i]].values[0:4])
        elif p[1] < 0.2 :
            submit_classes.extend(classes[classify_index[i]].values[0:1])
            submit_classes.append('new_whale')
            submit_classes.extend(classes[classify_index[i]].values[1:4])
        elif p[2] < 0.1 :
            submit_classes.extend(classes[classify_index[i]].values[0:2])
            submit_classes.append('new_whale')
            submit_classes.extend(classes[classify_index[i]].values[2:4])
        elif p[3] < 0.05 :
            submit_classes.extend(classes[classify_index[i]].values[0:3])
            submit_classes.append('new_whale')
            submit_classes.extend(classes[classify_index[i]].values[3:4])
        else:
            submit_classes.extend(classes[best_5_class[i]])
        classes_text = ' '.join(submit_classes)
        submit = submit.append(pd.Series(np.array([generator.filenames[i].split('/')[1], classes_text]), index=submit.columns), ignore_index=True)
    return submit

# load models
model = models.load_model('./kernels/v7/models/without_whalev7/56-4.45-3.24.hdf5')
# load test data
test_generator = load_test_data()
# predict
submit = predict(model, test_generator)

submit.to_csv('submit3.csv', index=False)
