from __future__ import division
import numpy as np
from tensorflow.keras.utils import to_categorical


from imageio import imread as imread
from imageio import imwrite as imsave
from skimage.transform import resize as imresize

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input



def get_img_raw(paths, Nbatch,h1=224,w1=224, color_type=3, normalize=True):
    if color_type == 1:
        imgs = np.zeros((Nbatch,h1,w1)).astype('float32')
    elif color_type == 3:
        imgs = np.zeros((Nbatch,h1,w1,3)).astype('float32')
    for i in range(Nbatch):
        path = paths[i]
        if color_type == 1:
            img = imread(path)
        elif color_type == 3:
            img = imread(path)
        resized = imresize(img, (h1,w1))
        resized = (resized*255).astype(np.uint8)
        imgs[i,:] = np.copy(resized)
        
    return imgs

def get_img(paths, Nbatch,h1=224,w1=224, color_type=3, normalize=True):
    if color_type == 1:
        imgs = np.zeros((Nbatch,h1,w1)).astype('float32')
    elif color_type == 3:
        imgs = np.zeros((Nbatch,h1,w1,3)).astype('float32')
    for i in range(Nbatch):
        path = paths[i]
        if color_type == 1:
            img = imread(path)
        elif color_type == 3:
            img = imread(path)
        resized = imresize(img, (h1,w1))
        resized = (resized*255).astype(np.uint8)
        if normalize:     
            resized = preprocess_input(resized)
        imgs[i,:] = np.copy(resized)
        
    return imgs


def get_train_multilabel(imgpath2,tokendata2,voc,h1,w1,Nbatch,Nstep,preprocess=False):
    while 1:
        i = 0 
        for j in range(0, Nstep):
            imgs = get_img(imgpath2[i:i+Nbatch], Nbatch,h1,w1,3,preprocess)
            token1 = np.tile(np.arange(voc, dtype='int32'), [Nbatch,1])
            token2 = tokendata2[i:i+Nbatch,0:voc]
            i = i+Nbatch
            yield([imgs,token1], {'lnow': token2})
def get_test_multilabel(imgpath2,tokendata2,voc,h1,w1,Nbatch,Nstep,i,preprocess=False):
            imgs = get_img(imgpath2[i:i+Nbatch], Nbatch,h1,w1,3,preprocess)
            token1 = np.tile(np.arange(voc, dtype='int32'), [Nbatch,1])
            token2 = tokendata2[i:i+Nbatch,0:voc]
            return imgs, token1, token2


def get_train_multilabelevidential(imgpath2,tokendata2,voc,h1,w1,Nbatch,Nstep,preprocess=False):
    while 1:
        i = 0 
        for j in range(0, Nstep):
            imgs = get_img(imgpath2[i:i+Nbatch], Nbatch,h1,w1,3,preprocess)
            token2 = tokendata2[i:i+Nbatch,0:voc]
            target = np.zeros((Nbatch,voc,1))
            i = i+Nbatch
            yield([imgs,token2], {'nll': target, 'kl': target})
def get_test_multilabelevidential(imgpath2,tokendata2,voc,h1,w1,Nbatch,Nstep,i,preprocess=False):
            imgs = get_img(imgpath2[i:i+Nbatch], Nbatch,h1,w1,3,preprocess)
            token2 = tokendata2[i:i+Nbatch,0:voc]
            target = np.zeros((Nbatch,voc,1))
            return imgs,token2, target

"""
def get_train_caption_mask(imgpath2,tokendata2,T,h1,w1,Nbatch,Nstep,preprocess=False):
    while 1:
        for i in range(0, len(imgpath2), Nbatch):
            imgs = get_img(imgpath2[i:i+Nbatch], Nbatch,h1,w1,3,preprocess)
            token1 = tokendata2[i:i+Nbatch,0:T-1]
            token2 = tokendata2[i:i+Nbatch,1:T]
            masks = token2>0
            yield([imgs,token1], {'lnow': token2}, masks)
def get_test_caption_mask(imgpath2,tokendata2,T,h1,w1,Nbatch,Nstep, i, preprocess=False):
            imgs = get_img(imgpath2[i:i+Nbatch], Nbatch,h1,w1,3,preprocess)
            token1 = tokendata2[i:i+Nbatch,0:T-1]
            token2 = tokendata2[i:i+Nbatch,1:T]
            masks = token2>0
            return imgs,token1, token2, masks
"""

def get_train_caption_mask(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,Nbatch,Nstep):
    while 1:
        for idx in range(0, Nstep):
            i = idx*Nbatch
            imgs = get_img(imgpath2[i:i+Nbatch], Nbatch)
            keymask = tokendatakey[i:i+Nbatch,0:Tkey]
            token1 = tokendata2[i:i+Nbatch,0:Tcap-1]
            token2 = tokendata2[i:i+Nbatch,1:Tcap]
            masks = token2>0
            target = np.zeros((Nbatch,Tcap-1,1))
            yield([imgs,token1,keymask,token2], {'nll':target,'kl':target}, masks)


def get_test_caption_mask(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,Nbatch,Nstep,i):
            imgs = get_img(imgpath2[i:i+Nbatch], Nbatch)
            keymask = tokendatakey[i:i+Nbatch,0:Tkey]
            token1 = tokendata2[i:i+Nbatch,0:Tcap-1]
            token2 = tokendata2[i:i+Nbatch,1:Tcap]
            masks = token2>0
            target = np.zeros((Nbatch,Tcap-1,1))
            return imgs,token1,keymask, token2, target,masks

def get_train_caption_mask_CrossEnt(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,Nbatch,Nstep):
    while 1:
        for idx in range(0, Nstep):
            i = idx*Nbatch
            imgs = get_img(imgpath2[i:i+Nbatch], Nbatch)
            keymask = tokendatakey[i:i+Nbatch,0:Tkey]
            token1 = tokendata2[i:i+Nbatch,0:Tcap-1]
            token2 = tokendata2[i:i+Nbatch,1:Tcap]
            masks = token2>0
            yield([imgs,token1,keymask], {'lnow':token2}, masks)
def get_test_caption_mask_CrossEnt(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,Nbatch,Nstep,i):
            imgs = get_img(imgpath2[i:i+Nbatch], Nbatch)
            imgs = get_img(imgpath2[i:i+Nbatch], Nbatch)
            keymask = tokendatakey[i:i+Nbatch,0:Tkey]
            token1 = tokendata2[i:i+Nbatch,0:Tcap-1]
            token2 = tokendata2[i:i+Nbatch,1:Tcap]
            masks = token2>0
            return imgs,token1,keymask, token2, masks
def get_train_caption_mask2_CrossEnt(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,Nbatch,Nstep,
                                     h1=299,preprocess=False):
    while 1:
        for idx in range(0, Nstep):
            i = idx*Nbatch
            imgs = get_img(imgpath2[i:i+Nbatch], Nbatch,h1,h1,3,preprocess) 
            keymask = tokendatakey[i:i+Nbatch,0:Tkey]
            token1 = tokendata2[i:i+Nbatch,0:Tcap-1]
            token2 = tokendata2[i:i+Nbatch,1:Tcap]
            
            imgs = np.concatenate([imgs,imgs], axis=0)
            token1 = np.concatenate([token1,token1], axis=0)
            token2 = np.concatenate([token2,token2], axis=0)
            keymask = np.concatenate([keymask,np.zeros(keymask.shape, dtype='int32')], axis=0)
            
            masks = (token2>0).astype('float32')
            masks[token2==5] = 0.2 
            yield([imgs,token1,keymask], {'lnow':token2}, masks)
def get_test_caption_mask2_CrossEnt(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,Nbatch,Nstep,i,
                                    h1=299,preprocess=False):
            imgs = get_img(imgpath2[i:i+Nbatch], Nbatch,h1,h1,3,preprocess) 
            keymask = tokendatakey[i:i+Nbatch,0:Tkey]
            token1 = tokendata2[i:i+Nbatch,0:Tcap-1]
            token2 = tokendata2[i:i+Nbatch,1:Tcap]
            
            imgs = np.concatenate([imgs,imgs], axis=0)
            token1 = np.concatenate([token1,token1], axis=0)
            token2 = np.concatenate([token2,token2], axis=0)
            keymask = np.concatenate([keymask,np.zeros(keymask.shape, dtype='int32')], axis=0)
            
            masks = (token2>0).astype('float32')
            masks[token2==5] = 0.2 
            return imgs,token1,keymask, token2, masks
