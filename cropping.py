import keras
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 

from PIL import Image 
from PIL.ImageDraw import Draw
from os.path import isfile

from matplotlib import pyplot as plt
import random
import numpy as np
from scipy.ndimage import affine_transform
from keras.preprocessing.image import img_to_array

from keras.models import Model
from keras import models

from keras.utils import Sequence
from keras import backend as K

image_dir = './data/test/0/'

def bounding_rectangle(list):
    x0, y0 = list[0]
    x1, y1 = x0, y0
    for x,y in list[1:]:
        x0 = min(x0, x)
        y0 = min(y0, y)
        x1 = max(x1, x)
        y1 = max(y1, y)
    return x0,y0,x1,y1

def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    rotation        = np.deg2rad(rotation)
    shear           = np.deg2rad(shear)
    rotation_matrix = np.array([[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shift_matrix    = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
    shear_matrix    = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix     = np.array([[1.0/height_zoom, 0, 0], [0, 1.0/width_zoom, 0], [0, 0, 1]])
    shift_matrix    = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))


anisotropy = 2.15
# Compute the coordinate transformation required to center the pictures, padding as required.
def center_transform(affine, input_shape):
    hi, wi = float(input_shape[0]), float(input_shape[1])
    ho, wo = float(128), float(128)
    top, left, bottom, right = 0, 0, hi, wi
    center_matrix   = np.array([[1, 0, -ho/2], [0, 1, -wo/2], [0, 0, 1]])
    scale_matrix    = np.array([[(bottom - top)/ho, 0, 0], [0, (right - left)/wo, 0], [0, 0, 1]])
    decenter_matrix = np.array([[1, 0, hi/2], [0, 1, wi/2], [0, 0, 1]])
    return np.dot(np.dot(decenter_matrix, scale_matrix), np.dot(affine, center_matrix))

# ### test
# image = read_raw_image(data[i][0])
# image = image.convert('L')
# m = np.array([[1,0,0],[0,1,0],[0,0,1]])
# m = center_transform(m, image.size)
# image = exec_transform(image, m)
# image = image.convert('L')
# show_image(np.array(image))
#
#### 

def coord_transform(list, trans):
    result = []
    for x,y in list:
        y,x,_ = trans.dot([y,x,1]).astype(np.int)
        result.append((x,y))
    return result

def gray_scale(image):
    image = image.convert('L')
    image = image.convert('RGB')
    return image

def get_transform_matrix(shape, validation):
    if validation:
        m = np.array([[1,0,0],[0,1,0],[0,0,1]])
    else:
        m  = build_transform(
                random.uniform(-5, 5),
                random.uniform(-5, 5),
                random.uniform(0.9, 1.0),
                random.uniform(0.9, 1.0),
                random.uniform(-0.05*input_size, 0.05*input_size),
                random.uniform(-0.05*input_size, 0.05*input_size))
    m = center_transform(m, shape)
    return m

def exec_transform(image, m):
    fill_color = int(np.average(image))
    image = image.convert('RGBA')
    image = image.transform((input_size, input_size), Image.AFFINE, m.reshape((9,1)), Image.BILINEAR)
    fff = Image.new('RGBA', image.size, (fill_color, fill_color, fill_color, 255))
    image = Image.composite(image, fff, image)
    return image

input_size = 128
seed = 1470
def preprocessing(image, coordinates, validation=False):
    image = gray_scale(image)
    m = get_transform_matrix(image.size, validation)
    image = exec_transform(image, m)
    image = image.convert('L')
    image = np.array(image) / 255
    image -= np.mean(image, keepdims=True)
    image /= np.std(image, keepdims=True) + K.epsilon()
    # drawbox
    transformed_coord = []
    if not validation:
        transformed_coord = [np.linalg.inv(m).dot([cc[0], cc[1], 1])[0:2] for cc in coordinates]
    # transformed_coord = coord_transform(coordinates, np.linalg.inv(m))
    # draw = Draw(image)
    # draw.rectangle(bounding_rectangle(transformed_coord), outline='red')
    return image, transformed_coord, m


def expand_path(p):
    return image_dir + p

def read_raw_image(p):
    return Image.open(expand_path(p))

model = models.load_model('cropping.hdf5')

files = os.listdir(image_dir)
for f in files:
    image = read_raw_image(f)
    new_image, trans, m= preprocessing(image, None, validation=True)
    rect = model.predict(new_image.reshape((1, 128, 128, 1)))[0]
    transformed_coord = [m.dot([cc[0], cc[1], 1])[0:2] for cc in [(rect[0], rect[1]), (rect[2], rect[3])]]
    margin = 10
    transformed_coord = [max(0, transformed_coord[0][0]-margin), max(0, transformed_coord[0][1]-margin), min(image.size[0], margin+transformed_coord[1][0]), min(image.size[1], margin+transformed_coord[1][1])]
    image.crop(transformed_coord).save('./data/test_cropped/0/%s' %(f))
    # image.show()
    # show_predict(image, transformed_coord)

