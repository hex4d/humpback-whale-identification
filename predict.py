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

def load_data():
    datagen = ImageDataGenerator(
       rescale=1./255,
    )
    train_generator = datagen.flow_from_directory(
        'data/train',
        target_size=(input_size, input_size),
        batch_size=1,
        shuffle=False,
        seed=seed,
    )
    return train_generator

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

def save_class():
    train_generator = load_data()
    classes= train_generator.class_indices
    df = pd.DataFrame.from_dict(classes, orient='index')
    df.to_csv('train_classes_classify.cls')

def predict(new_known_model, classify_model, generator, steps):
    num = 5
    generator.reset()
    pred_new = new_known_model.predict_generator(generator, steps=steps, verbose=1)
    generator.reset()
    pred_classify = classify_model.predict_generator(generator, steps=steps, verbose=1)
    classify_index = np.argpartition(-pred_classify, num)[:, :num]
    classify_value = pred_classify[np.arange(pred_classify.shape[0])[:, None], classify_index]
    # create output
    df = pd.read_csv('train_classes_classify.cls')
    classes = df.loc[:, 'Unnamed: 0']
    submit = pd.DataFrame(columns=['Image', 'Id'])
    for i, pred in enumerate(pred_new):
        sort_index = np.argsort(classify_value[i])[::-1]
        classify_index[i] = classify_index[i][sort_index]
        classify_value[i] = classify_value[i][sort_index]
        submit_classes = []
        # if pred[0] > 0.9999 and classify_value[i][0] < 0.5:
        #     submit_classes.append('new_whale')
        #     submit_classes.extend(classes[classify_index[i]].values[0:4])
        # elif pred[0] > 0.9:
        #     submit_classes.extend(classes[classify_index[i]].values[0:4])
        #     submit_classes.append('new_whale')
        # else:
        submit_classes.extend(classes[classify_index[i]].values[0:5])
        classes_text = ' '.join(submit_classes)
        submit = submit.append(pd.Series(np.array([generator.filenames[i].split('/')[1], classes_text]), index=submit.columns), ignore_index=True)
    return submit

# load models
new_known_model = models.load_model('models/separate1_new_known/model.h5')
classify_model = models.load_model('models/separate2_classify/115-4.96-4.36.hdf5')
# load test data
test_generator = load_test_data()
train_generator = load_data()
# predict
submit = predict(new_known_model, classify_model, test_generator, steps=len(test_generator))

submit.to_csv('submit.csv', index=False)



