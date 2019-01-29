import os
import subprocess
import shutil
import pandas as pd
import glob

CLASS_COUNT_LIMIT = 5
VALIDATION_SPLIT = 0.1
SEED = 1470

BASE_DIR = 'data'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')

def download():
    if not os.path.exists('data'):
        os.mkdir('data')
    os.chdir('data')
    if not os.path.exists('train.zip'):
        subprocess.call(['kaggle', 'competitions', 'download', '-c', 'humpback-whale-identification'])
    shutil.rmtree('test')
    os.mkdir('test')
    os.chdir('test')
    subprocess.call(['unzip', '../test.zip', '-d', '0'])
    os.chdir('../')
    subprocess.call(['unzip', 'train.zip', '-d', 'train'])

def train_validation_split(class_list, validation_split):
    # 5 class 未満は全てtrain
    class_count = class_list.count()[0]
    if class_count < CLASS_COUNT_LIMIT:
        return class_list, pd.DataFrame(columns=class_list.columns)
    # 1 or n * validation_split
    validation_count = int(max(class_count * validation_split, 1))
    train_count = class_count - validation_count
    validation_list = class_list.sample(validation_count, random_state=SEED)
    train_list = class_list.drop(validation_list.index)
    return train_list, validation_list

def initialize_train_validation(clean = False):
    # make train and validation data
    # nc = nc > 5 && nc else 0 
    # validation_count = max( 1, nc/10 )
    classes = pd.read_csv('data/train.csv')
    class_value_counts = classes['Id'].value_counts().reset_index()
    # 
    if clean:
        path = os.path.join('./' + TRAIN_DIR, '**/*.txt')
        [shutil.move(x, TRAIN_DIR) for x in glob.glob(path, recursive=True)]
        path = os.path.join('./' + VALIDATION_DIR, '**/*.txt')
        [shutil.move(x, TRAIN_DIR) for x in glob.glob(path, recursive=True)]
        glob.glob(path, recursive=True)
        [shutil.rmtree(os.path.join(TRAIN_DIR, x)) for x in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, x))]
        [shutil.rmtree(os.path.join(VALIDATION_DIR, x)) for x in os.listdir(VALIDATION_DIR) if os.path.isdir(os.path.join(VALIDATION_DIR, x))]
    if not os.path.exists(TRAIN_DIR):
        os.mkdir(TRAIN_DIR)
    if not os.path.exists(VALIDATION_DIR):
        os.mkdir(VALIDATION_DIR)
    for i, v in class_value_counts.iterrows():
        class_name = v[0]
        train_output_dir = os.path.join(TRAIN_DIR, class_name)
        if not os.path.exists(train_output_dir):
            os.mkdir(train_output_dir)
        validation_output_dir = os.path.join(VALIDATION_DIR, class_name)
        if not os.path.exists(validation_output_dir):
            os.mkdir(validation_output_dir)
        class_files_list = classes[classes['Id'] == class_name]
        train_list, validation_list = train_validation_split(class_files_list, VALIDATION_SPLIT)
        [shutil.move(os.path.join(TRAIN_DIR, x[1]['Image']), validation_output_dir) for x in validation_list.iterrows()]
        [shutil.move(os.path.join(TRAIN_DIR, x[1]['Image']), train_output_dir) for x in train_list.iterrows()]


download()
initialize_train_validation(False)

assert(len(os.listdir('./data/validation')) == len(os.listdir('./data/train')))
