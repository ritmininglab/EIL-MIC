import heapq
import numpy as np
from PIL import ImageTk, Image
import tkinter as tk
import matplotlib.pyplot as plt
from config import Tcap, Tkey
from imageio import imread
from skimage.transform import resize as imresize
from nltkeval import calculate_meteor
import tensorflow as tf
from config import N, h1, w1, Nbatch, Tkey, Tcap, voc2, weightadj, dictappend
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import EfficientNetB1 as CNN
from tensorflow.keras.models import Model

adamlarge = tf.keras.optimizers.Adam(
    learning_rate=0.0025, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
adam = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
adamsmall = tf.keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
adamtiny = tf.keras.optimizers.Adam(
    learning_rate=0.00005, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
adamnano = tf.keras.optimizers.Adam(
    learning_rate=0.00002, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)

class heapNode:
    def __init__(self, val, data, mask):
        self.val = val
        self.data = data
        self.mask = mask
    def __eq__(self, other):
        return self.val==other.val
    def __lt__(self, other):
        return self.val<other.val
    def __gt__(self, other):
        return self.val>other.val

def visualizedata(img2):
    if np.max(img2) > 1:
        img2 = img2.astype('uint8')
    plt.figure(figsize=(20, 20))
    plt.imshow(img2)
    plt.axis('off')
    plt.show()
    plt.clf()


def visualizeraw(i, imgpath):
    img = imread(imgpath[i])
    resized = imresize(img, (224, 224))
    plt.figure(figsize=(20, 20))
    plt.imshow(resized)
    plt.axis('off')
    plt.show()
    plt.clf()


def translate(i, dictionary, prednum, printstr='pred', fprint=True):
    Tkey = prednum.shape[1]
    matchnow = prednum[i, 0:Tkey]
    words = ''
    for t in range(Tcap-1):
        wordidx = matchnow[t]
        if wordidx == 2:  
            break
        words += dictionary[wordidx]+' '
    if fprint:
        print(str(i)+' '+printstr+': '+words)
    return words


def translateKey(i, dictionary, prednum, printstr='pred', fprint=True):
    Tkey = prednum.shape[1]
    matchnow = prednum[i, 0:Tkey]
    words = ''
    for t in range(Tkey):
        if matchnow[t] != 0:
            words += dictionary[t]+' '
    if fprint:
        print(str(i)+' '+printstr+': '+words)
    return words

def userKeywords2token(userstr, Tk, dictwordidx):
    useranno = userstr.split(" ")
    tokens = np.zeros((1, Tk)).astype('int32')
    for i in range(len(useranno)):
        word = useranno[i]
        if word not in dictwordidx:
            print(f'Not in dictionary: {word}')
        else:
            tokens[0, dictwordidx[word]] = 1
    return tokens



ximg = Input(shape=(h1, w1, 3), name='xcnn')
xkey = Input(shape=(Tkey), name='xkey', dtype='int32')
xcap = Input(shape=(Tcap-1), name='xcap')
xtar = Input(shape=(Tcap-1), name='xtar', dtype='int32')
xre = Input(shape=(Tcap-1, voc2), name='xre', dtype='float32')
xunpair = Input(shape=(1, 1), name='xp', dtype='int32')

hw = 10
dimimg = 1280
xcnn = Input(shape=(h1, w1, 3), name='xcnn')
cnn = CNN(include_top=False, weights="imagenet", input_tensor=xcnn)
cnn.trainable = False

ximg = Input(shape=(h1,w1, 3), name='xcnn')
xtxtk = Input(shape=(Tkey), name='xtxtk', dtype='int32')
xtarget = Input(shape=(Tkey), name='xtarget', dtype='int32')
xrek = Input(shape=(Tkey, 2), name='xre', dtype='float32')

from utilModel import ViTcaptionSoftmax as BFvggk2
from utilModel import ViT as multilabel
from utilModel import ViTcaptionVgg2unpair as BFvggevi

def get_cap():
    sparselabel = False
    dropout = 0.2
    dimsk = [256, 1280, Tkey, voc2, Tcap-1, sparselabel]  
    bnparamk = [1e-5, [True, True], dropout]

    m2 = Model(inputs=[ximg, xcap, xkey, xre, xunpair],
              outputs=BFvggk2([ximg, xcap, xkey, xre, xunpair], dimsk, bnparamk, cnn))
    m2.compile(loss={'lnow': 'categorical_crossentropy'},
              optimizer=adamsmall,  
              weighted_metrics={'lnow': 'categorical_crossentropy'})  
    return m2
def get_capevi():
    sparselabel = False
    dropout = 0.2
    dimsk = [256, 1280, Tkey, voc2, Tcap-1, sparselabel] 
    bnparamk = [1e-5,[True,True],dropout]
    
    m2 = Model(inputs=[ximg, xcap, xkey, xre, xunpair],
              outputs=BFvggevi([ximg, xcap, xkey, xre, xunpair], dimsk, bnparamk, cnn))
    m2.compile(loss={'nll': 'mae',  'kl': 'mae'},
              loss_weights={'nll': 1.0, 'kl': 1/1000},
              optimizer=adamsmall,
              weighted_metrics={'nll': 'mae',  'kl': 'mae'}) 
    return m2
def get_key():
    sparselabel = False
    dimsk = [512, 256, Tkey,sparselabel] 
    bnparamk = [1e-5,[True,True]]
    m0 = Model(inputs=[ximg, xtxtk, xrek],
              outputs=multilabel([ximg, xtxtk, xrek], dimsk, bnparamk, cnn, useevi=True))
    m0.compile(loss={'nll': 'mae',  'kl': 'mae'},
              loss_weights={'nll': 1.0, 'kl': 0.5}, 
              optimizer=adamsmall,
              metrics={'nll': 'mae',  'kl': 'mae'}) 
    return m0

"""
ximg = Input(shape=(h1, w1, 3), name='xcnn')
xkey = Input(shape=(Tkey), name='xkey', dtype='int32')
xcap = Input(shape=(Tcap-1), name='xcap')
xtar = Input(shape=(Tcap-1), name='xtar', dtype='int32')
xre = Input(shape=(Tcap-1, voc2), name='xre', dtype='float32')
xunpair = Input(shape=(1, 1), name='xp', dtype='int32')

hw = 10
dimimg = 1280
xcnn = Input(shape=(h1, w1, 3), name='xcnn')
cnn = CNN(include_top=False, weights="imagenet", input_tensor=xcnn)
cnn.trainable = False



sparselabel = False
dropout = 0.2
dimsk = [256, 1280, Tkey, voc2, Tcap-1, sparselabel]  
bnparamk = [1e-5, [True, True], dropout]


m2 = Model(inputs=[ximg, xcap, xkey, xre, xunpair],
          outputs=BFvggk2([ximg, xcap, xkey, xre, xunpair], dimsk, bnparamk, cnn))
m2.compile(loss={'lnow': 'categorical_crossentropy'},
          optimizer=adamnano,  
          weighted_metrics={'lnow': 'categorical_crossentropy'})  
teacher = Model(inputs=[ximg, xcap, xkey, xre, xunpair],
          outputs=BFvggk2([ximg, xcap, xkey, xre, xunpair], dimsk, bnparamk, cnn))
teacher.compile(loss={'lnow': 'categorical_crossentropy'},
          optimizer=adamnano,  
          weighted_metrics={'lnow': 'categorical_crossentropy'})  
"""

def predkey(m0, imgs,token1, token2, target, dictionary):
    token2hot = tf.keras.utils.to_categorical(token2,2)
    
    preds = m0.predict([imgs,token1, token2hot*0], batch_size=10)
    predsbinary = preds[0][:,:,1] > preds[0][:,:,0]
    for i in range(2): 
        translateKey(i, dictionary, token2, 'truth')
        translateKey(i, dictionary, predsbinary, 'predi')
    
    hit = (predsbinary * token2)==1
    prec = np.sum(hit) / (np.sum(predsbinary)+1e-5)
    recall = np.sum(hit) / np.sum(token2)
    f1 = 2*prec*recall / (prec+recall+1e-5)

    evi = preds[0] 
    S = np.sum(evi+1, axis=-1, keepdims=True)
    vac = 2/S
    diss = 2*np.amin(evi/S, axis=-1, keepdims=True)
    
    pnow = 0.8
    ewc = pnow*np.squeeze(vac) + (pnow-0.5)*np.squeeze(diss)
    
    print(f"vac={np.mean(vac)} diss={np.mean(diss)} ewc={np.mean(ewc)}")
    return preds,vac,diss,ewc,prec,recall,f1

def selectkey(preds,vac,diss,ewc,token2, usehard):
    predsbinary = preds[0][:,:,1] > preds[0][:,:,0]
    Ntest = predsbinary.shape[0]
    top_k = 8
    anno_mask = np.zeros_like(predsbinary)
    for i in range(Ntest):
        poslist = set(np.where(predsbinary[i,:]==True)[0].tolist())
        count = len(poslist)
        descending = np.argsort(-ewc[i,:])
        for j in descending:
            if j not in poslist:
                poslist.add(j)
                count += 1
                if count==top_k:
                    break
        for j in poslist:
            anno_mask[i,j] = 1
    ewc2 = 0.7 - 0.7*np.squeeze(vac) - (0.7-0.5)*np.squeeze(diss)
    mask3now = ewc2*(1-anno_mask) + anno_mask
    
    evi = preds[0] 
    S = np.sum(evi+1, axis=-1, keepdims=True)
    softlabelkey = (evi+1) / S
    gt1hot = tf.keras.utils.to_categorical(token2, 2)
    temp = anno_mask[:,:,np.newaxis]
    softlabelkey = softlabelkey*(1-temp) + gt1hot*temp
    if usehard:
        softlabelkey = np.argmax(softlabelkey, axis=-1)
        softlabelkey = tf.keras.utils.to_categorical(softlabelkey, 2)
    return mask3now,anno_mask,softlabelkey



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

def updateKeywordPred(pred, list1, ans2idx):
    for i in range(len(list1)):
        if list1[i]==0:
            pred[0,ans2idx[i]] = 0
        else:
            pred[0,ans2idx[i]] = 1
    return pred
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


def beam_evi_too_slow(m2,imgnow,keynow,token1now,token2now,dictionaryAll, topk=3,fprint=False):
    temp1 = np.zeros(token1now.shape, dtype='int32')
    wordidx = 1 
    temp1[:,0] = wordidx
    temp2 = np.zeros(token2now.shape, dtype="float32")
    token2nowhot =  tf.keras.utils.to_categorical(token2now,voc2)
    unpairnow = np.ones((1,1,1), dtype="int32")
    
    caps1 = [temp1]
    scores1 = [0]
    
    beam = 3  
    caps2 = [] 
    scores2 = [] 
    caps3 = [] 
    scores3 = [] 

    topk = []
    heapq.heapify(topk)
    replay = []
    heapq.heapify(replay)
    
    for t in range(Tcap-1):
        for tempnow, scorenow in zip(caps1, scores1):
            preds = m2.predict([imgnow,tempnow,keynow, token2nowhot*0, unpairnow])
            evi = preds[0][0,t,:] 
            S = np.sum(evi+1, axis=-1, keepdims=True)
            softlabel = (evi+1)/S
            logit = np.log(softlabel+1e-8)
            lls = logit - np.max(logit)
            lls = logit - np.log( np.sum(np.exp(logit)))
            ranked = np.argsort(-lls)
            for j in range(beam):
                word = ranked[j]
                tempnew = tempnow.copy()
                if t<Tcap-2:
                    tempnew[0,t+1] = word
                scorenew = scorenow+np.log(softlabel[word]+1e-8)
                if word==2 or t==Tcap-2:
                    heapq.heappush(topk, heapNode(-scorenew, tempnew))
                else:
                    caps2.append(tempnew)
                    scores2.append(scorenew)
        
        if len(caps2)==0 or len(topk)==beam*4:
            break
            
        caps1 = [] 
        scores1 = [] 
        ranked = np.argsort(-np.array(scores2))
        for j in range(min(beam, len(caps2))):
            caps1.append(caps2[ranked[j]])
            scores1.append(scores2[ranked[j]])
        caps2 = []
        scores2 = []
            
    gt = translate(0, dictionaryAll, token2now, 'targe', fprint=fprint)
    record = [x for x in topk]
    for j in range(min(beam*3, len(topk))):
        data = heapq.heappop(topk)
        pred = translate(0, dictionaryAll, data.data, f'beam {data.val}', fprint=fprint)
        meteor = calculate_meteor(pred, [gt])
        heapq.heappush(replay, heapNode(-meteor,data.data))
    return replay


def beambatch_evi(m2,imgnow0,keynow0,token1now0,token2now0,dictionaryAll, topk=3,fprint=False):
    temp1 = np.zeros(token1now0.shape, dtype='int32')
    wordidx = 1 
    temp1[:,0] = wordidx
    temp2 = np.zeros(token2now0.shape, dtype="float32")
    token2nowhot0 =  tf.keras.utils.to_categorical(token2now0,voc2)
    unpairnow0 = np.ones((1,1,1), dtype="int32")
    
    caps1 = [temp1]
    scores1 = [0]
    
    beam = 3  
    caps2 = [] 
    scores2 = [] 
    caps3 = [] 
    scores3 = [] 

    topk = []
    heapq.heapify(topk)
    replay = []
    heapq.heapify(replay)
    
    for t in range(Tcap-1):
        tempnow = np.concatenate(caps1, axis=0)
        nbatch = tempnow.shape[0]
        imgnow = np.tile(imgnow0, [nbatch,1,1,1])
        keynow = np.tile(keynow0, [nbatch,1])
        token2nowhot = np.tile(token2nowhot0, [nbatch,1,1])
        unpairnow = np.tile(unpairnow0, [nbatch,1,1])
        preds = m2.predict([imgnow,tempnow,keynow, token2nowhot*0, unpairnow])
        for i in range(len(caps1)):
            scorenow = scores1[i]
            evi = preds[0][i,t,:] 
            S = np.sum(evi+1, axis=-1, keepdims=True)
            softlabel = (evi+1)/S
            logit = np.log(softlabel+1e-8)
            lls = logit - np.max(logit)
            lls = logit - np.log( np.sum(np.exp(logit)))
            ranked = np.argsort(-lls)
            for j in range(beam):
                word = ranked[j]
                tempnew = tempnow[i:i+1].copy()
                if t<Tcap-2:
                    tempnew[0,t+1] = word
                scorenew = scorenow+np.log(softlabel[word]+1e-8)
                if word==2 or t==Tcap-2:
                    heapq.heappush(topk, heapNode(-scorenew, tempnew, None))
                else:
                    caps2.append(tempnew)
                    scores2.append(scorenew)
        
        if len(caps2)==0 or len(topk)==beam*3:
            break
            
        caps1 = [] 
        scores1 = [] 
        ranked = np.argsort(-np.array(scores2))
        for j in range(min(beam, len(caps2))):
            caps1.append(caps2[ranked[j]])
            scores1.append(scores2[ranked[j]])
        caps2 = []
        scores2 = []
            
    gt = translate(0, dictionaryAll, token2now0, 'targe', fprint=fprint)
    record = [x for x in topk]
    for j in range(min(beam*3, len(topk))):
        data = heapq.heappop(topk)
        pred = translate(0, dictionaryAll, data.data, f'beam {data.val}', fprint=fprint)
        meteor = calculate_meteor(pred, [gt])
        heapq.heappush(replay, heapNode(-meteor,data.data, None))
    return replay
