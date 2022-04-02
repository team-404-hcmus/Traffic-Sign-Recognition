
# import os, pickle, shutil
# import numpy as np
# from skimage.io import imread
# import skimage.morphology as morp
# from skimage.filters import rank
# from sklearn.utils import shuffle, compute_class_weight
# from sklearn.metrics import confusion_matrix
# import csv
# import cv2
# import matplotlib.pyplot as plt
# from keras.models import Input, Model
# from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
# from tensorflow.keras.utils import to_categorical
# from keras import optimizers
# from keras.initializers import random_normal
# from keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint
# import seaborn as sn
# from sklearn.metrics import confusion_matrix

import zipfile

import zipfile
with zipfile.ZipFile("./data/traffic-signs-data.zip", 'r') as zip_ref:
    zip_ref.extractall("./data")