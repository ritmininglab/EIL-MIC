
import tensorflow as tf

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
import tkinter as tk
from PIL import ImageTk, Image
from imageio.v2 import imread,imwrite
import cv2



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
with open('auxiliary/TagDict.tsv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        dictionary.append(row[1])
dictionaryAll = []
with open('auxiliary/WordDict.tsv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        dictionaryAll.append(row[1])

def visualizedata(img2):
    if np.max(img2) > 1:
        img2 = img2.astype('uint8')
    plt.figure(figsize=(20, 20))
    plt.imshow(img2)
    plt.axis('off')
    plt.show()
def visualizeraw(i,imgpath):
    img = imread(imgpath[i])
    resized = imresize(img, (224,224))
    plt.figure(figsize=(20, 20))
    plt.imshow(resized)
    plt.axis('off')
    plt.show()
def translate(i, dictionary, prednum, printstr='pred'):
    Tkey = prednum.shape[1]
    matchnow = prednum[i, 0:Tkey]
    words = ''
    for t in range(Tkey):
        wordidx = matchnow[t]
        if wordidx == 2:
            break
        words += dictionary[wordidx]+' '
    print('Image '+str(i)+' '+printstr+': '+words)

def translateKey(i, dictionary, prednum, printstr='pred'):
    Tkey = prednum.shape[1]
    matchnow = prednum[i, 0:Tkey]
    words = ''
    for t in range(Tkey):
        if matchnow[t] != 0:
            words += dictionary[t]+' '
    print('Image '+str(i)+' '+printstr+': '+words)

N = 7439 
h1 = 299 
w1 = 299 
Nbatch = 20 
Tkey = 50 



from utilInterpret import loaddata1key,loaddata2key
from utilInterpret import loaddata1caption
imgpath = loaddata1key('auxiliary/imgpath.csv', 'resizedimg/', N)
tokendata = loaddata2key('auxiliary/imgtag.csv', Tkey, N)
metaefficient = 1*np.ones((N,), dtype='int8')

Tcap = 57 
voc2 = 1284 
from utilInterpret import loaddata1caption
capdata = loaddata1caption('auxiliary/imgtoken.csv', Tcap, N)



mode = 1
if mode==1:

    import random
    myorder = np.arange(N).tolist()
    random.Random(0).shuffle(myorder)
    imgpath3 = [imgpath[i] for i in myorder]
    tokendata3 = tokendata[myorder,]
    capdata3 = capdata[myorder,]
    
    bias = 0
    imgpath2 = imgpath3[bias:bias+N]
    tokendata2 = tokendata3[bias:bias+N]
    capdata2 = capdata3[bias:bias+N]







hw = 10 
dimimg = 1280 
Nstep = int(N/Nbatch)

from tensorflow.keras.applications import EfficientNetB1 as CNN
xcnn = Input(shape=(h1,w1, 3), name='xcnn')
cnn = CNN(include_top=False, weights="imagenet", input_tensor=xcnn)
cnn.trainable = False




from utilIOkey import get_train_multilabelevidential as get_train
from utilIOkey import get_test_multilabelevidential as get_test

from utilIOkey import get_train_caption_mask2_CrossEnt as get_train3
from utilIOkey import get_test_caption_mask2_CrossEnt as get_test3

mode = 0
if mode == 0:
    from imageio import imread as imread
    from imageio import imwrite as imsave
    from skimage.transform import resize as imresize






def mylosscatgorical(y_true, y_pred):
    temp = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=False, axis=-1)
    return temp
def mylossbinary(y_true, y_pred):
    temp = tf.keras.losses.binary_crossentropy(
        y_true, y_pred, from_logits=False)
    return temp
def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, tf.cast(y_pred>0.5, dtype='float32')), axis=-1)

dimsk = [512, [64,128,256,512], Tkey] 
bnparamk = [1e-5,[True,True]]
ximg = Input(shape=(h1,w1, 3), name='xcnn')
xtxtk = Input(shape=(Tkey), name='xtxtk', dtype='int32')

from mCap3 import ViTevi as multilabel

m0 = Model(inputs=[ximg, xtxtk],
          outputs=multilabel([ximg, xtxtk], dimsk, bnparamk, cnn))
m0.compile(loss={'nll': 'mae',  'kl': 'mae'},
          loss_weights={'nll': 1.0, 'kl': 0.05},
          optimizer=adam,
          metrics={'nll': 'mae',  'kl': 'mae'}) 

Nstep = 250 
mygenerator = get_train(imgpath2,tokendata2,Tkey,h1,w1,Nbatch,Nstep,False)
    
m0.load_weights('auxiliary/m3.h5')


ximg = Input(shape=(h1,w1, 3), name='xcnn')
xkey = Input(shape=(Tkey), name='xkey', dtype='int32')
xcap = Input(shape=(Tcap-1), name='xcap') 
xtar = Input(shape=(Tcap-1), name='xtar', dtype='int32')
from mCap3 import ViTcaptionVgg2 as BFvggk2

dimsk = [512, 1280, Tkey, voc2, Tcap-1] 
bnparamk = [1e-5,[True,True]]

m2 = Model(inputs=[ximg, xcap, xkey],
          outputs=BFvggk2([ximg, xcap, xkey], dimsk, bnparamk, cnn))


Nstep = 250 
mygenerator2 = get_train3(imgpath2,tokendata2,capdata2, Tkey, Tcap, Nbatch,Nstep)
m2.load_weights('auxiliary/m4.h5')




def logit2binary(logits):
    binary = logits[:,:,0] < logits[:,:,1] 
    binary = binary.astype('int8')
    return binary
def evaluateUnc2(evi, acc):
    weight = np.zeros(evi.shape, dtype='float32')
    
    opinion = evi+1
    nclass = evi.shape[-1]
    S = np.sum(opinion, axis=-1)
    vac = nclass / S
    bs = evi / np.expand_dims(S, axis=-1)
    diss = 2*np.min(bs, axis=-1)
    wv = acc/(2*acc-0.5)
    weight = wv*vac + (1-wv)*diss
    return vac, diss, weight
def translateQuery(weightnow, dictionary, K=8):
    orderedidx = np.argsort(weightnow) 
    ans = []
    answeight = []
    for k in range(K):
        ans.append(dictionary[orderedidx[-k-1]])
        answeight.append(weightnow[orderedidx[-k-1]])
    return ans, answeight, orderedidx
dictwordidx = {}
for j in range(len(dictionary)):
    word = dictionary[j]
    dictwordidx[word] = j
def selectQuery(ans, predsbinary, dictionary,dictwordidx, K1=2, K=8):
    ans2 = []
    ans2idx = []
    K1 = min(K1, np.sum(predsbinary==1))
    K2 = K-K1
    keyidx = np.where(predsbinary==1)[0] 
    for k in range(K1):
        ans2.append(dictionary[keyidx[k]])
        ans2idx.append(keyidx[k])
    remain = K2
    for k in range(K):
        word = ans[k]
        if word not in ans2:
            ans2.append(word)
            ans2idx.append(dictwordidx[word])
            remain -= 1
            if remain==0:
                break
    return ans2, ans2idx
def guiSelectKey(ans2):
    filepath = 'test/0.gif'
    root = tk.Tk()
    root.title("GUI")
    root.geometry('500x620+100+50')
    canvas = tk.Canvas(root, width=460,height=370, bd=0, highlightthickness=0)
    img = Image.open(filepath)
    photo = ImageTk.PhotoImage(img) 
    canvas.create_image(230,170,image=photo)
    canvas.create_text(30, 350, text="Select keywords:", fill="black", 
                       anchor=tk.NW, width=400,
                       font=('Helvetica 10 bold'))
    canvas.pack()
    
    K = 8
    list1 = [0]*K
    listvar = []
    for _ in range(K):
        listvar.append(tk.IntVar())
    def print_selection():
        for k in range(K):
            list1[k] = listvar[k].get()
    def closewindow():
        root.destroy()
    listc = []
    for k in range(K):
        listc.append(tk.Checkbutton(root, text=ans2[k],variable=listvar[k], 
                                  onvalue=1,offvalue=0, command=print_selection))
        listc[k].pack()
    button_calc = tk.Button(root, text="Submit",
                            command=closewindow)
    button_calc.pack()
    root.mainloop()
    return list1

mode = 0
if mode==0: 

    Ntest = 1
    bias = 5009
    imgs,token2, target  = get_test(imgpath2,tokendata2,Tkey,h1,w1,Ntest,Nstep,bias,False)
    
    preds = m0.predict([imgs,token2], batch_size=5)
    predsbinary = logit2binary(preds[0])
    for i in range(Ntest):
        translateKey(i, dictionary, token2, 'truth')
        translateKey(i, dictionary, predsbinary, 'predi')
        visualizedata(imgs[i]) 
        
    evi = preds[0]
    vac, diss, weight = evaluateUnc2(evi, 0.8)

    i = 0
    weightnow = weight[i]
    
    ans, answeight, orderedidx = translateQuery(weightnow, dictionary, K=16)
    ans2, ans2idx = selectQuery(ans, predsbinary[i], dictionary,dictwordidx, K1=4, K=8)
    
    filepath = 'test/0.gif'
    res = cv2.imread(imgpath2[bias+i])
    res = cv2.resize(res, dsize=(320,320))
    res2 = np.zeros(res.shape,dtype='uint8')
    for j in range(3):
        res2[:,:,j] = res[:,:,2-j]
    imwrite(filepath,res2) 
    
    list1 = guiSelectKey(ans2)
    
    pred = predsbinary[i:i+1]
    def updateKeywordPred(pred, list1, ans2idx):
        for i in range(len(list1)):
            if list1[i]==0:
                pred[0,ans2idx[i]] = 0
            else:
                pred[0,ans2idx[i]] = 1
        return pred
    pred = updateKeywordPred(pred, list1, ans2idx)
        
        





    
def guiInput():
    def getwords():
        global userstr
        userstr = entry.get()
    def closewindow():
        root.destroy()
    filepath = 'test/0.gif'
    root = tk.Tk()
    root.title("GUI")
    root.geometry('512x512+50+50')
    canvas = tk.Canvas(root, width=512,height=450, bd=0, highlightthickness=0)
    img = Image.open(filepath)
    photo = ImageTk.PhotoImage(img) 
    canvas.create_image(256,200,image=photo)
    canvas.pack()
    entry = tk.Entry(root, insertbackground='white',highlightthickness=2)
    entry.pack()
    button_calc = tk.Button(root, text="Specify keywords and submit",
                            command=lambda: [getwords(), closewindow()])
    button_calc.pack()
    root.mainloop()
    return userstr
def guiOutput2(predi,truth):
    def closewindow():
        root.destroy()
    filepath = 'test/0.gif'
    root = tk.Tk()
    root.title("GUI")
    root.geometry('512x512+50+50')
    canvas = tk.Canvas(root, width=512,height=490, bd=0, highlightthickness=0)
    img = Image.open(filepath)
    photo = ImageTk.PhotoImage(img)
    canvas.create_image(256,160,image=photo)
    canvas.create_text(50, 350, text="Caption:", fill="black", 
                       anchor=tk.NW, width=512-80,
                       font=('Helvetica 10 bold'))
    canvas.create_text(50, 370, text=predi, fill="black", 
                       anchor=tk.NW, width=512-80,
                       font=('Helvetica 10'))
    canvas.create_text(50, 400, text="Groud truth:", fill="black", 
                       anchor=tk.NW, width=512-80,
                       font=('Helvetica 10 bold'))
    canvas.create_text(50, 420, text=truth, fill="black", 
                       anchor=tk.NW, width=512-80,
                       font=('Helvetica 10'))
    canvas.pack()
    button_calc = tk.Button(root, text="Exit",
                            command=closewindow)
    button_calc.pack()
    root.mainloop()


mode = -1
if mode==0:
    Ntest = 10
    bias = 2000
    imgs2,token1cap,keytoken,token2cap, masks \
        = get_test3(imgpath2,tokendata2,capdata2, Tkey,Tcap, Ntest,Nstep,bias)
    preds = m2.predict([imgs2,token1cap,keytoken])
    predcap = np.argmax(preds, axis=-1)
    for i in range(Ntest):
        translate(i, dictionaryAll, token2cap, 'targe')
        translate(i, dictionaryAll, predcap, 'predi')
        print('')
        visualizedata(imgs2[i])
        
elif mode==1: 
    Ntest = 10
    bias = 5000
    imgs2,token1cap,keytoken,token2cap, masks \
        = get_test3(imgpath2,tokendata2,capdata2, Tkey,Tcap, Ntest,Nstep,bias)
    for i in range(Ntest*1):
        imgnow = imgs2[i:i+1]
        token1now = token1cap[i:i+1]
        token2now = token2cap[i:i+1]
        keynow = keytoken[i:i+1]
        temp = np.zeros(token1now.shape, dtype='int32')
        wordidx = 1 
        for t in range(Tcap-1):
            temp[:,t] = wordidx
            if wordidx==2:
                break
            preds = m2.predict([imgnow,temp,keynow])
            wordidx = np.argmax(preds[0,t,:], axis=-1)
        predcap = temp[:,1:]
        translateKey(i, dictionary, keytoken, 'key')
        translate(0, dictionaryAll, token2now, 'targe')
        translate(0, dictionaryAll, predcap, 'predi')
        print('')
        visualizedata(np.clip(imgs2[i], 0, 255))



def topKcandidate3update(candidates3,scores3,temp2,temp2score, beam):
    if len(candidates3)<beam:
        candidates3.append(temp2)
        scores3.append(temp2score)
    else:
        scores3np = np.asarray(scores3)
        minscore = np.min(scores3np)
        if temp2score>minscore: 
            minidx = np.where(scores3np==minscore)[0][0]
            candidates3[minidx] = temp2
            scores3[minidx] = temp2score
    return candidates3, scores3

def topKcandidate3rank(candidates3,scores3,beam):
    candidates4 = []
    scores4 = []
    order = np.argsort(scores3)
    for i in range(len(order)):
        pos = order[len(order) - i-1]
        candidates4.append(candidates3[pos])
        scores4.append(scores3[pos])
    return candidates4, scores4

def translatebeam1(i, dictionary, candidates3, scores3, printing=False):
    Tkey = candidates3[0].shape[1]
    beam = len(candidates3)
    result = []
    for i in range(beam):
        matchnow = candidates3[i]
        words = ''
        for t in range(Tkey):
            wordidx = matchnow[0,t]
            if wordidx >3:
                words += dictionary[wordidx]+' '
        if printing:
            print(words)
        result.append(words)
    return result


def guiSelectCap(ans3):
    filepath = 'test/0.gif'
    root = tk.Tk()
    root.title("GUI")
    root.geometry('500x680+100+50')
    canvas = tk.Canvas(root, width=460,height=570, bd=0, highlightthickness=0)
    img = Image.open(filepath)
    photo = ImageTk.PhotoImage(img) 
    canvas.create_image(230,170,image=photo)
    canvas.create_text(20, 345, text="Predictions:", fill="black", 
                       anchor=tk.NW, width=420,
                       font=('Helvetica 10 bold'))
    K = 3
    textnow = ''
    for k in range(K):
        textnow = textnow + str(k)+': '+ans3[k] + '\n\n'
    canvas.create_text(20, 360, text=textnow, fill="black", 
                       anchor=tk.NW, width=420,
                       font=('Helvetica 10'))
    canvas.create_text(20, 550, text="Select:", fill="black", 
                       anchor=tk.NW, width=420,
                       font=('Helvetica 10 bold'))
    canvas.pack()
    
    list1 = [0]*K
    listvar = []
    for _ in range(K):
        listvar.append(tk.IntVar())
    def print_selection():
        for k in range(K):
            list1[k] = listvar[k].get()
    def closewindow():
        root.destroy()
    listc = []
    for k in range(K):
        listc.append(tk.Checkbutton(root, text=str(k), variable=listvar[k], 
                                  onvalue=1,offvalue=0, command=print_selection))
        listc[k].pack()
    button_calc = tk.Button(root, text="Submit",
                            command=closewindow)
    button_calc.pack()
    root.mainloop()
    return list1

mode = 1
if mode==1:
    
    Ntest = 1
    bias = 5009
    imgs2,token1cap,keytoken,token2cap, masks \
        = get_test3(imgpath2,tokendata2,capdata2, Tkey,Tcap, Ntest,Nstep,bias)
      
        
    keytoken = pred
    for i in range(1): 
        beam = 2
        beamshow = 3
        candidates1 = np.zeros((beam, Tcap-1), dtype='int32')
        candidates2 = np.zeros((beam*beam, Tcap-1), dtype='int32')
        scores1 = -1000*np.ones((beam,))
        scores2 = -1000*np.zeros((beam*beam))
        candidates3 = [] 
        scores3 = []
    
        imgnow = imgs2[i:i+1]
        token1now = token1cap[i:i+1]
        token2now = token2cap[i:i+1]
        keynow = keytoken[i:i+1]
    
        candidates1[:, 0] = 1
        t = 0
        temp = candidates1[0:1]
        preds = m2.predict([imgnow,token1now,keynow])

        lls = np.log(preds[0,t,:]+1e-8)
        ranked = np.argsort(-lls)
        for j in range(beam):
            candidates1[j, t+1] = ranked[j]
            scores1[j] = lls[ranked[j]]
        
        validcandi1 = beam
        for t in range(1, Tcap-2):
            validcandi2 = np.zeros((beam,), dtype='int8')
            scores2 = -1000*np.ones((beam*beam))
            for k in range(min(beam,validcandi1)):
                temp = candidates1[k:k+1]
                preds = m2.predict([imgnow,temp,keynow])

                lls = np.log(preds[0,t,:]+1e-8)
                ranked = np.argsort(-lls)
                for j in range(beam):
                    word = ranked[j]
                    temp2 = np.copy(temp)
                    temp2[0,t+1] = word
                    temp2score = scores1[k] + lls[ranked[j]]
                    if word==2: 
                        candidates3,scores3 = topKcandidate3update(candidates3,scores3,temp2,temp2score, beam*2)
                    else: 
                        idxcandi = np.sum(validcandi2)
                        candidates2[idxcandi] = temp2
                        scores2[idxcandi] = temp2score
                        validcandi2[k] = validcandi2[k] + 1
            
            validcandi1 = np.sum(validcandi2>0)
            if validcandi1==0: 
                break
            ranked2 = np.argsort(-scores2)
            for k in range(min(beam, np.sum(validcandi2))):
                candidates1[k] = candidates2[ranked2[k]]
                scores1[k] = scores2[ranked2[k]]
        
    candidates4, scores4 = topKcandidate3rank(candidates3,scores3,beam)
    cut = min(6,beamshow)
    
    print('')
    print('Automated prediction of key concepts:')
    ans3 = translatebeam1(i, dictionaryAll, candidates4[:cut], scores4[:cut], 
                            printing=False)

    list2 = guiSelectCap(ans3)


