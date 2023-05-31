import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

path = ['test/0.jpg','test/1.jpg']
dicpath = 'test/Dictionary.csv'

class CustomCallbackpretrain(Callback):
    def on_train_begin(self, logs={}):    
        self.epochs = 0    
    def on_epoch_end(self, batch, logs={}):    
        self.epochs += 1     
        if self.epochs % 10 != 11:     
            print("epoch {}, loss {:3.3f}={:3.3f}+{:3.3f} acc {:1.3f}".format(
                self.epochs, logs["loss"], logs["lnow_loss"], logs["loss"]-logs["lnow_loss"],
                logs["lnow_accuracy"])
                )
class CustomCallback(Callback):
    def on_train_begin(self, logs={}):    
        self.epochs = 0    
    def on_epoch_end(self, batch, logs={}):    
        self.epochs += 1     
        if self.epochs % 20 == 1:     
            print("epoch {}, loss {:3.3f}={:3.3f}+{:3.3f} acc {:1.3f}".format(
                self.epochs, logs["loss"], logs["lnow_loss"], logs["loss"]-logs["lnow_loss"],
                logs["lnow_accuracy"])
                )
class CustomCallbackkernel(Callback):
    def on_train_begin(self, logs={}):    
        self.epochs = 0    
    def on_epoch_end(self, batch, logs={}):    
        self.epochs += 1     
        if self.epochs % 10 == 1:     
            print("epoch {}, loss {:3.3f}".format(
                self.epochs, logs["loss"])
                )
class CustomCallbackmrf(Callback):
    def on_train_begin(self, logs={}):    
        self.epochs = 0    
    def on_epoch_end(self, batch, logs={}):    
        self.epochs += 1     
        if self.epochs % 100 == 1:     
            print("epoch {}, loss {:3.3f}={:3.3f}+{:3.3f} acc {:1.3f}".format(
                self.epochs, logs["loss"], logs["lnow_loss"], logs["loss"]-logs["lnow_loss"],
                logs["lnow_accuracy"])
                )
def myloss(y_true, y_pred):
    temp = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_true,logits=y_pred)
    return temp
def mylosssparse(y_true, y_pred):
    temp = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true,logits=y_pred)
    return temp
def mylosssoftmaxsparse(y_true, y_pred):
    temp = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=False, axis=-1)
    return temp
def mylosssoftmax(y_true, y_pred):
    temp = tf.keras.losses.categorical_crossentropy(
        y_true, y_pred, from_logits=True)
    return temp

def configdims(dimdata, dimclass, d1,d2,d3):
    d4 = d2+d3
    dims = [[dimdata,d1,d2,d2,d2,d2,d2,d3, d4,d3,d4,d3,d4,d3,d3],
        [d1,d2,d2,d2,d2,d2,d3,d3,d3,d3,d3,d3,d3,d3,dimclass]]
    return dims

def passweights(mfrom, mto):
    weights = mfrom.get_weights()
    mto.set_weights(weights)
    return mto

def getWcls(clstruth, dimcls):
    result = np.zeros((dimcls,))
    for cc in range(dimcls):
        result[cc] = np.sum(clstruth==cc)
    result = result/np.sum(result)
    return result

def getMRFparams(apred,bpred):
    a0 = apred[0].astype(np.float32)
    b0 = (np.log(bpred[0,:,:,0:1])+8).astype(np.float32)
    c0 = [0.5,1e-3]
    return [a0,b0,c0]

def ioulogitAggregate(clstruth, clspred, wcls, taskid=999, string='unknown'):
    ypred = np.argmax(clspred, axis=-1)
    ytrue = np.argmax(clstruth, axis=-1)
    iou = 0
    for cc in range(0,wcls.shape[0]):
        intersect = np.sum((ypred==cc)*(ytrue==cc))
        union = np.sum(((ypred==cc)+(ytrue==cc)) > 0)
        if union>0:
            iou += intersect / union * wcls[cc]
    
    return iou

def ioulogitIndividual(clstruth, clspred, wcls, taskid=999, string='unknown'):
    ypred = np.argmax(clspred, axis=-1)
    ytrue = np.argmax(clstruth, axis=-1)
    dimcls = len(wcls)
    iou = 0
    for cc in range(0,dimcls):
        intersect = np.sum((ypred==cc)*(ytrue==cc))
        union = np.sum(((ypred==cc)+(ytrue==cc)) > 0)
        if union>0:
            iou += intersect / union * np.sum(ytrue==cc)
    iou = iou / (ytrue.shape[-1]*ytrue.shape[-2])
    
    return iou

def npsigmoid(x):
    y = 1/(1 + np.exp(-x)) 
    return y
def npdesigmoid(x, clipvalue):
    xclip = np.clip(x, clipvalue, 1-clipvalue)
    y = np.log(xclip/(1-xclip))
    return y
def clipsigmoid(x, clipthreshold):
    xclip = np.clip(x, -clipthreshold, clipthreshold)
    y = 1/(1 + np.exp(-xclip)) 
    return y

def lgt2ent(x):
    x = np.exp(x-np.max(x)) + 1e-6
    y = 1/np.sum(x) * x
    z = -np.sum(y*np.log(y))
    return z

def checkresult(forward0, msg):
    temp = 0
    for i in range(len(forward0)-3,len(forward0)):
        temp += forward0[i]
    print(msg, " unseen task iou = ", temp/3)
    
