from __future__ import division
import numpy as np


from imageio import imread as imread
from skimage.transform import resize as imresize

import tensorflow as tf

from tensorflow.keras.applications.vgg16 import preprocess_input


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


def get_test_multilabelevidential(imgpath2,tokendata2,voc,h1,w1,Nbatch,Nstep,i,preprocess=False):
            imgs = get_img(imgpath2[i:i+Nbatch], Nbatch,h1,w1,3,preprocess)
            token1 = np.tile(np.arange(voc, dtype='int32'), [Nbatch,1])
            token2 = tokendata2[i:i+Nbatch,0:voc]
            target = np.zeros((Nbatch,voc,1))
            return imgs,token1,token2, target
def get_train_multilabelevidential(imgpath2,tokendata2,voc,h1,w1,Nbatch,Nstep,preprocess=False):
    while 1:
        i = 0 
        for j in range(0, Nstep):
            imgs,token1,token2, target = get_test_multilabelevidential(imgpath2,tokendata2,voc,h1,w1,Nbatch,Nstep,i,preprocess)
            i = i+Nbatch
            yield([imgs,token1,token2], {'nll': target, 'kl': target})


def re_train_keymask_mixed(imgpath2,tokendata2,voc,h1,w1,Nbatch,Nstep,
                           imgpath3,tokendata3,mask3,preprocess=False):
    
    Nbatch3 = tokendata3.shape[0]//Nstep if tokendata3 is not None else 0
    while 1:
        i = 0 
        i2 = 0
        for j in range(0, Nstep):
            imgs = get_img(imgpath2[i:i+Nbatch], Nbatch,h1,w1,3,preprocess)
            token2 = tf.keras.utils.to_categorical(tokendata2[i:i+Nbatch,0:voc],2)
            mask = np.ones((token2.shape[0], token2.shape[1]))
            if tokendata3 is not None:
                imgs2 = get_img(imgpath3[i2:i2+Nbatch3], Nbatch3,h1,w1,3,preprocess)
                token22 = tokendata3[i2:i2+Nbatch3,0:voc]
                mask2 = mask3[i2:i2+Nbatch3]
                imgs = np.concatenate([imgs, imgs2], axis=0)
                token2 = np.concatenate([token2, token22], axis=0)
                mask = np.concatenate([mask, mask2], axis=0)
            token1 = np.tile(np.arange(voc, dtype='int32'), [token2.shape[0],1])
            target = np.zeros((token2.shape[0],voc,1))
            i += Nbatch
            i2 += Nbatch3
            yield([imgs,token1,token2], {'nll':target,'kl':target}, {'nll':mask,'kl':mask})

def get_test_caption_mask2_CrossEnt(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,Nbatch,Nstep,i,
                                    h1=299,preprocess=False,weightadj=[5]):
            imgs = get_img(imgpath2[i:i+Nbatch], Nbatch,h1,h1,3,preprocess) 
            keymask = tokendatakey[i:i+Nbatch,0:Tkey]
            token1 = tokendata2[i:i+Nbatch,0:Tcap-1]
            token2 = tokendata2[i:i+Nbatch,1:Tcap]
            
            imgs = np.concatenate([imgs,imgs], axis=0)
            token1 = np.concatenate([token1,token1], axis=0)
            token2 = np.concatenate([token2,token2], axis=0)
            keymask = np.concatenate([keymask,np.zeros(keymask.shape, dtype='int32')], axis=0)
            
            masks = (token2>0).astype('float32')
            for adj in weightadj:
                masks[token2==adj] = 0.5 
            return imgs,token1,keymask, token2, masks

def get_train_caption_mask2_evi(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,Nbatch,Nstep,
                                h1=299,preprocess=False,weightadj=[5]): 
    while 1:
        i = 0 
        for j in range(0, Nstep):
            imgs,token1,keymask, token2, masks = get_test_caption_mask2_CrossEnt(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,
                                                             Nbatch,Nstep,i,h1,preprocess)
            target = np.zeros((Nbatch*2,Tcap-1,1))
            masks = masks[:,:,np.newaxis]
            i = i+Nbatch
            yield([imgs,token1,keymask,token2], {'nll':target,'kl':target}, {'nll':masks,'kl':masks})


def retrain_caption_mask2_evi(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,Nbatch,Nstep,
                              imgpath3,tokendata3a,tokendata3b,Nstep3,voc2,h1=299,preprocess=False,weightadj=[5]): 
    while 1:
        i = 0 
        for j in range(0, Nstep+Nstep3):
            if j<Nstep:
                imgs,token1,keymask,token2,masks = get_test_caption_mask2_CrossEnt(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,
                                                                 Nbatch,Nstep,i,h1,preprocess)
                token2 = tf.keras.utils.to_categorical(token2,voc2)
            else:
                i2 = i-Nstep*Nbatch
                imgs = get_img(imgpath3[i2:i2+Nbatch], Nbatch,h1,h1,3,preprocess)
                keymask = tokendatakey[i:i+Nbatch,0:Tkey]
                token1 = tokendata3a[i2:i2+Nbatch,:]
                token2 = tokendata3b[i2:i2+Nbatch,0:] 
                token2num = np.zeros_like(token1)
                token2num[:,0:Tcap-2] = token1[:,1:Tcap-1]
                token2 = tf.keras.utils.to_categorical(token2num,voc2)
                
                imgs = np.concatenate([imgs,imgs], axis=0)
                token1 = np.concatenate([token1,token1], axis=0)
                token2 = np.concatenate([token2,token2], axis=0)
                keymask = np.concatenate([keymask,np.zeros(keymask.shape, dtype='int32')], axis=0)
                token2num = np.concatenate([token2num,token2num], axis=0)
                
                masks = (token2num>0).astype('float32')
                for adj in weightadj:
                    masks[token2num==adj] = 0.5 
                    
            target = np.zeros((Nbatch*2,Tcap-1,1))
            masks = masks[:,:,np.newaxis]
            i = i+Nbatch
            yield([imgs,token1,keymask,token2], {'nll':target,'kl':target}, {'nll':masks,'kl':masks})


def get_test_caption_unpair(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,Nbatch,Nstep,i,
                            h1=299,preprocess=False,weightadj=[5],unpairshift=3488,noimg=False):
            if noimg:
                imgs = np.zeros((Nbatch,1,1,3))
            else:
                imgs = get_img(imgpath2[i:i+Nbatch], Nbatch,h1,h1,3,preprocess) 
            keymask = tokendatakey[i:i+Nbatch,0:Tkey]
            token1 = tokendata2[i:i+Nbatch,0:Tcap-1]
            token2 = tokendata2[i:i+Nbatch,1:Tcap]
            unpair = np.ones((imgs.shape[0],1,1))
            if unpairshift is not None:
                imgs0 = np.zeros_like(imgs) 
                keymask3 = tokendatakey[unpairshift+i:unpairshift+i+Nbatch,0:Tkey]
                token13 = tokendata2[unpairshift+i:unpairshift+i+Nbatch,0:Tcap-1]
                token23 = tokendata2[unpairshift+i:unpairshift+i+Nbatch,1:Tcap]
                unpair0 = np.zeros((imgs.shape[0],1,1))
                
                imgs = np.concatenate([imgs,imgs0,imgs0,imgs,imgs0,imgs0], axis=0)
                token1 = np.concatenate([token1,token1,token13,token1,token1,token13], axis=0)
                token2 = np.concatenate([token2,token2,token23,token2,token2,token23], axis=0)
                keymask = np.concatenate([keymask,keymask,keymask3,np.zeros_like(keymask,dtype='int32'),
                                          np.zeros_like(keymask,dtype='int32'),np.zeros_like(keymask,dtype='int32')], axis=0)
                unpair = np.concatenate([unpair,unpair0,unpair0,unpair,unpair0,unpair0], axis=0)
            else:
                imgs = np.concatenate([imgs,imgs], axis=0)
                token1 = np.concatenate([token1,token1], axis=0)
                token2 = np.concatenate([token2,token2], axis=0)
                keymask = np.concatenate([keymask,np.zeros_like(keymask,dtype='int32')], axis=0)
                unpair = np.concatenate([unpair,unpair], axis=0)
            masks = (token2>0).astype('float32')
            for adj in weightadj:
                masks[token2==adj] = 0.5 
            masks = masks * (0.95*unpair[:,:,0] + 0.05)
            return imgs,token1,keymask, token2, masks, unpair


def pretrain_caption_softlabel_softmax(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,Nbatch,Nstep,
                            imgpath3,tokendata3a,tokendata3b,Nstep3,voc2,h1=299,preprocess=False,weightadj=[5],unpairshift=3488,bias=0): 
    while 1:
        i = bias 
        for j in range(0, Nstep+Nstep3):
            if j<Nstep:
                imgs,token1,keymask,token2,masks,unpair = get_test_caption_unpair(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,
                                                                 Nbatch,Nstep,i,h1,preprocess,weightadj,unpairshift)
                token2 = tf.keras.utils.to_categorical(token2,voc2)
            i = i+Nbatch
            yield([imgs,token1,keymask,token2,unpair], {'lnow':token2}, {'lnow':masks})

def retrain_softmax_mixed(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,Nbatch,Nstep,
                         imgpath3,tokendatakey3,tokendata3,voc2,h1=299,preprocess=False,weightadj=[5]): 
    while 1:
        i = 0 
        i3 = 0
        Nbatch3 = tokendata3.shape[0]//Nstep
        for j in range(0, Nstep):
            imgs,token1,keymask,token2,masks,unpair = get_test_caption_unpair(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,
                                                             Nbatch,Nstep,i,h1,preprocess,weightadj,unpairshift=None)
            token2 = tf.keras.utils.to_categorical(token2,voc2)
            i = i+Nbatch
            imgs3,token13,keymask3,token23,masks3,unpair3 = get_test_caption_unpair(imgpath3,tokendatakey3,tokendata3,Tkey,Tcap,
                                                             Nbatch3,Nstep,i3,h1,preprocess,weightadj,unpairshift=None)
            token23 = tf.keras.utils.to_categorical(token23,voc2)
            i3 = i3+Nbatch3
            imgs = np.concatenate([imgs,imgs3], axis=0)
            token1 = np.concatenate([token1,token13], axis=0)
            keymask = np.concatenate([keymask,keymask3], axis=0)
            token2 = np.concatenate([token2,token23], axis=0)
            masks = np.concatenate([masks,masks3*0.2], axis=0)
            unpair = np.concatenate([unpair,unpair3], axis=0)
            yield([imgs,token1,keymask,token2,unpair], {'lnow':token2}, {'lnow':masks})
def retrain_softmax_mixed_soft(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,Nbatch,Nstep,
                         imgpath3,tokendatakey3,tokendata3,capdatasoft,voc2,h1=299,preprocess=False,weightadj=[5]): 
    while 1:
        i = 0 
        i3 = 0
        Nbatch3 = tokendata3.shape[0]//Nstep
        for j in range(0, Nstep):
            imgs,token1,keymask,token2,masks,unpair = get_test_caption_unpair(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,
                                                             Nbatch,Nstep,i,h1,preprocess,weightadj,unpairshift=None)
            token2 = tf.keras.utils.to_categorical(token2,voc2)
            i = i+Nbatch
            imgs3,token13,keymask3,token23,masks3,unpair3 = get_test_caption_unpair(imgpath3,tokendatakey3,tokendata3,Tkey,Tcap,
                                                             Nbatch3,Nstep,i3,h1,preprocess,weightadj,unpairshift=None)
            token23 = np.concatenate([capdatasoft[i3:i3+Nbatch3],capdatasoft[i3:i3+Nbatch3]],axis=0)
            i3 = i3+Nbatch3
            
            imgs = np.concatenate([imgs,imgs3], axis=0)
            token1 = np.concatenate([token1,token13], axis=0)
            keymask = np.concatenate([keymask,keymask3], axis=0)
            token2 = np.concatenate([token2,token23], axis=0)
            masks = np.concatenate([masks,masks3*1], axis=0)
            unpair = np.concatenate([unpair,unpair3], axis=0)
            yield([imgs,token1,keymask,token2,unpair], {'lnow':token2}, {'lnow':masks})

def check(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,Nbatch,Nstep,
                         imgpath3,tokendatakey3,tokendata3,voc2,h1=299,preprocess=False,weightadj=[5]): 
    T=2
    i = 0 
    i3 = 0
    for t in range(T):

        Nbatch3 = tokendata3.shape[0]//Nstep
        imgs,token1,keymask,token2,masks,unpair = get_test_caption_unpair(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,
                                                         Nbatch,Nstep,i,h1,preprocess,weightadj,unpairshift=None)
        i = i+Nbatch
        imgs3,token13,keymask3,token23,masks3,unpair3 = get_test_caption_unpair(imgpath3,tokendatakey3,tokendata3,Tkey,Tcap,
                                                         Nbatch3,Nstep,i3,h1,preprocess,weightadj,unpairshift=None)
        i3 = i3+Nbatch3
        
        imgs = np.concatenate([imgs,imgs3], axis=0)
        token1 = np.concatenate([token1,token13], axis=0)
        keymask = np.concatenate([keymask,keymask3], axis=0)
        token2 = np.concatenate([token2,token23], axis=0)
        masks = np.concatenate([masks,masks3*0.2], axis=0)
        unpair = np.concatenate([unpair,unpair3], axis=0)
    return imgs,token1,keymask,token2,unpair,masks,i,i3



def retrain_caption_evi_mixed(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,Nbatch,Nstep,
                           imgpath3,tokendatakey3,tokendata3a,voc2,h1=299,preprocess=False,weightadj=[],unpairshift=None): 
    while 1:
        i = 0 
        Nbatch3 = tokendata3a.shape[0]//Nstep
        i2 = 0
        for j in range(0, Nstep):
            imgs,token1,keymask,token2,masks,unpair = get_test_caption_unpair(imgpath2,tokendatakey,tokendata2,Tkey,Tcap,
                                                             Nbatch,Nstep,i,h1,preprocess,weightadj,unpairshift=None)
            imgs3,token13,keymask3,token23,masks3,unpair3 = get_test_caption_unpair(imgpath3,tokendatakey3,tokendata3a,Tkey,Tcap,
                                                             Nbatch3,Nstep,i2,h1,preprocess,weightadj,unpairshift=None)
            imgs = np.concatenate([imgs, imgs3], axis=0)
            token1 = np.concatenate([token1, token13], axis=0)
            keymask = np.concatenate([keymask, keymask3], axis=0)
            token2 = np.concatenate([token2, token23], axis=0)
            masks = np.concatenate([masks, masks3*0.5], axis=0) 
            unpair = np.concatenate([unpair, unpair3], axis=0)
            token2 = tf.keras.utils.to_categorical(token2,voc2)
                    
            target = np.zeros((imgs.shape[0],Tcap-1,1))
            masks = masks[:,:,np.newaxis]
            i = i+Nbatch
            i2 = i2+Nbatch3
            yield([imgs,token1,keymask,token2,unpair], {'nll':target,'kl':target}, {'nll':masks,'kl':masks})
