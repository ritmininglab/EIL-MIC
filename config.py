import numpy as np
import tensorflow as tf

from utilMisc import configdims

h1 = 256
w1 = 352
N = 4800 
Nbatch = 4
N2 = 8
dimdata = 3
dimcls = 2
kldiv = 20*h1*w1
kldiv2 = 1000*h1*w1
Ncore = 63
Ncorebatch = Nbatch


lW = ['b1a','b1b','b2a','b2b','b3a','b3b','b4a','b4b','b5a','b5b','b6a','b6b','b7a','b7b']
lZ = ['b1az','b1bz','b2az','b2bz','b3az','b3bz','b4az','b4bz','b5az','b5bz','b6az','b6bz','b7az','b7bz']
lWlnow = ['b1a','b1b','b2a','b2b','b3a','b3b','b4a','b4b','b5a','b5b','b6a','b6b','b7a','b7b','lnow']

dims = configdims(dimdata, dimcls, 20,40,35)
dims2 = configdims(dimdata, dimcls, 36,60,55)

trainables = []
for i in range(len(dims[0])):
    trainables.append(True)

p1 = 'script/0.h5'
p2 = 'script/1.h5'
p3 = 'script/2.h5'
p4 = 'script/3.h5'

h2 = 64
w2 = 88
dimsvae = [20,30,30,30,30,30,30,30,1000,1000,1000,16*22*30,30,30,30,30,30,30,30,20]
paramsvae = [1, h2*w2*3, [16,22,30]]


adamlarge2 = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
adamlarge = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
adamsmall = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
adamsmall2 = tf.keras.optimizers.Adam(learning_rate=0.00025, beta_1=0.9, beta_2=0.99, epsilon=1e-06,)
sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
sgd0step = tf.keras.optimizers.SGD(learning_rate=0.)
verbose = 0


