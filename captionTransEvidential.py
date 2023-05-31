"""
x = tf.Variable(tf.random.uniform([5, 30], -1, 1))
s = tf.split(x, num_or_size_splits=3, axis=1)

x = tf.Variable(tf.random.uniform([1,4,4,3], -1, 1))
s = tf.compat.v1.nn.max_pool(x, ksize=(2,2), strides=(2,2), padding='VALID')
"""

import tensorflow as tf
"""
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


import re
from tensorflow.keras.callbacks import Callback
from matplotlib.patches import Rectangle
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from matplotlib import pyplot
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from tensorflow.keras import backend as K
from skimage.transform import resize
import os

import scipy.io
import scipy.misc
import numpy as np
import math
import csv




class CustomCallback(Callback):
    def on_train_begin(self, logs={}):
        self.epochs = 0

    def on_epoch_end(self, batch, logs={}):
        self.epochs += 1
        if self.epochs % 2 == 1:
            print("epoch {}, loss {:3.3f}={:3.3f}+{:3.3f} metric {:1.3f}".format(
                self.epochs, logs["loss"], logs["lnow_loss"], logs["loss"] -
                    logs["lnow_loss"],
                logs["lnow_accuracy"])
                )


seed = 111
np.random.seed(seed)
tf.random.set_seed(seed)

adamlarge = tf.keras.optimizers.Adam(
    learning_rate=0.0025, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
adam = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
adamsmall = tf.keras.optimizers.Adam(
    learning_rate=0.0005, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
sgd0step = tf.keras.optimizers.SGD(learning_rate=0.)
verbose = 0





dictionary = []
with open('TagDict.tsv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        dictionary.append(row[-1])
dictionaryAll = []
with open('WordDict.tsv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        dictionaryAll.append(row[-1])

def visualizedata(img2):
    if np.max(img2) > 1:
        img2 = img2.astype('uint8')
    plt.figure(figsize=(20, 20))
    plt.imshow(img2)
    plt.axis('off')
 
