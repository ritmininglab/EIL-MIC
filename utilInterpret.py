import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def logging(history, m2=None, folder="folder",msg="msg",note=None, txtfile = "log.txt"):
    isExist = os.path.exists(folder)
    if not isExist:
       os.makedirs(folder)
       print("The new directory is created!")
       
    if m2 is not None:
        m2.save_weights(f'{folder}/{msg}.h5')
    now = datetime.now()
    numpy_loss_history = np.array(history.history['loss'])
    with open(folder+"/"+txtfile, 'a') as file1:
        file1.write(f"{now}:\n")
        file1.write(f"    msg: {folder}/{msg}.h5\n")
        if note is not None:
            file1.write(f"    note: {note}\n")
        file1.write(f"    train_loss: {np.array2string(numpy_loss_history)}\n")
        if 'val_loss' in history.history.keys():
            file1.write(f"    val_loss: {np.array2string(np.array(history.history['val_loss']))}\n")
    print(history.history.keys())
    
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history.keys():
        plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f"{folder}/loss {msg}.png")
    plt.show()
    plt.clf()


def logging2(history, logging2interval=2,logging2heatup=8, folder="folder",msg="msg",note=None, txtfile = "log2.txt"):
    isExist = os.path.exists(folder)
    if not isExist:
       os.makedirs(folder)
       print("The new directory is created!")
       
    now = datetime.now()
    numpy_bleu_history = np.array(history['bleu'])
    numpy_rouge_history = np.array(history['rouge'])
    numpy_meteor_history = np.array(history['meteor'])
    with open(folder+"/"+txtfile, 'a') as file1:
        file1.write(f"{now}:\n")
        file1.write(f"    msg: {folder}/{msg}.h5\n")
        if note is not None:
            file1.write(f"    note: {note}\n")
        file1.write(f"    train_loss: {np.array2string(numpy_bleu_history)}\n")
        file1.write(f"    train_loss: {np.array2string(numpy_rouge_history)}\n")
        file1.write(f"    train_loss: {np.array2string(numpy_meteor_history)}\n")
    
    plt.plot(history['bleu'])
    plt.plot(history['rouge'])
    plt.plot(history['meteor'])
    plt.title(f'validation nltk {msg}')
    plt.ylabel('nltk')
    plt.xlabel('epoch')
    plt.legend(['blue', 'rouge', 'meteor'], loc='upper left')
    xticks_default = range(len(history['bleu']))
    xticks = [logging2interval*i+logging2heatup for i in xticks_default]
    plt.xticks(xticks_default, xticks)
    plt.savefig(f"{folder}/nltk {msg}.png")
    plt.show()
    plt.clf()

def calculate_ma(weightsma, weightsnew, decay=0.8):
    newma = []
    for i in range(len(weightsma)):
        newlayer = weightsma[i]*decay + weightsnew[i]*(1-decay)
        newma.append(newlayer)
    return newma

from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from config import Tcap, voc2
from nltkeval import calculate_bleu, calculate_rouge, calculate_meteor


def greedy_batch(m2, imgs2b, token1capb, keytokenb, token2capb, masksb, Ntest2, Nbatch, usekey=True,earlystop=1e4):
    output = []
    for idx in tqdm(range(min(Ntest2//Nbatch, earlystop))):
        i = idx*Nbatch
        imgnow = imgs2b[i:i+Nbatch]
        token1now = token1capb[i:i+Nbatch]
        token2now = token2capb[i:i+Nbatch]
        token2nowhot = to_categorical(token2now, num_classes=voc2, dtype='float32')
        keynow = keytokenb[i:i+Nbatch]
        if not usekey:
            keynow = 0*keynow
        unpairnow = np.ones((Nbatch, 1, 1), dtype="int32")
        temp = np.zeros(token1now.shape, dtype='int32')
        wordidx = 1  
        count_eos = np.zeros((Nbatch), dtype="int32")
        pos_eos = (Tcap-1)*np.ones((Nbatch), dtype="int32")
        for t in range(Tcap-1):
            temp[:, t] = wordidx
            if np.sum(count_eos) == Nbatch:
                break
            preds = m2.predict([imgnow, temp, keynow, token2nowhot*0, unpairnow],verbose=None)
            wordidx = np.argmax(preds[0][:, t, :], axis=-1)
            count_eos = np.logical_or(count_eos, wordidx == 2)
            pos_eos = np.minimum(pos_eos, count_eos*t+(1-count_eos)*Tcap)
            """
            if wordidx==2 and t<5:
                ordered = np.argsort(-preds[0][0,t,:])
                print(f"{i}:{temp[0]} {ordered[0]}")
                assert ordered[0]==2, f"{i}:{temp[0]} {ordered[0]} should be 2"
                wordidx = ordered[1]
            """
        predcap = temp 
        
        for j in range(Nbatch):
            predcap[j,pos_eos[j]+2:] = 0
        
        output.append(predcap)
    output = np.concatenate(output, axis=0)
    return output
def greedy_batch_soft(m2, imgs2b, token1capb, keytokenb, token2capb, masksb, Ntest2, Nbatch, usekey=True,earlystop=1e4):
    output = []
    outputsoft = []
    for idx in tqdm(range(min(Ntest2//Nbatch, earlystop))):
        i = idx*Nbatch
        imgnow = imgs2b[i:i+Nbatch]
        token1now = token1capb[i:i+Nbatch]
        token2now = token2capb[i:i+Nbatch]
        token2nowhot = to_categorical(token2now, num_classes=voc2, dtype='float32')
        keynow = keytokenb[i:i+Nbatch]
        if not usekey:
            keynow = 0*keynow
        unpairnow = np.ones((Nbatch, 1, 1), dtype="int32")
        temp = np.zeros(token1now.shape, dtype='int32')
        tempsoft = np.zeros_like(token2nowhot, dtype='float32')
        wordidx = 1  
        count_eos = np.zeros((Nbatch), dtype="int32")
        pos_eos = (Tcap-1)*np.ones((Nbatch), dtype="int32")
        for t in range(Tcap-1):
            temp[:, t] = wordidx
            if np.sum(count_eos) == Nbatch:
                break
            preds = m2.predict([imgnow, temp, keynow, token2nowhot*0, unpairnow],verbose=None)
            wordidx = np.argmax(preds[0][:, t, :], axis=-1)
            count_eos = np.logical_or(count_eos, wordidx == 2)
            pos_eos = np.minimum(pos_eos, count_eos*t+(1-count_eos)*Tcap)
            tempsoft[:,t,:] =  preds[0][:, t, :]
            """
            if wordidx==2 and t<5:
                ordered = np.argsort(-preds[0][0,t,:])
                print(f"{i}:{temp[0]} {ordered[0]}")
                assert ordered[0]==2, f"{i}:{temp[0]} {ordered[0]} should be 2"
                wordidx = ordered[1]
            """
        predcap = temp 
        for j in range(Nbatch):
            predcap[j,pos_eos[j]+2:] = 0
        output.append(predcap)
        outputsoft.append(tempsoft)
    output = np.concatenate(output, axis=0)
    outputsoft = np.concatenate(outputsoft, axis=0)
    return output, outputsoft
def greedy_batch_adjust_logit(m2, imgs2b, token1capb, keytokenb, token2capb, masksb,
                            Ntest2, Nbatch, usekey=True,earlystop=1e4):
    output = []
    for idx in tqdm(range(min(Ntest2//Nbatch, earlystop))):
        i = idx*Nbatch
        imgnow = imgs2b[i:i+Nbatch]
        token1now = token1capb[i:i+Nbatch]
        token2now = token2capb[i:i+Nbatch]
        token2nowhot = to_categorical(token2now, num_classes=voc2, dtype='float32')
        keynow = keytokenb[i:i+Nbatch]
        if not usekey:
            keynow = 0*keynow
        unpairnow = np.ones((Nbatch, 1, 1), dtype="int32")
        temp = np.zeros(token1now.shape, dtype='int32')
        wordidx = 1  
        count_eos = np.zeros((Nbatch), dtype="int32")
        pos_eos = (Tcap-1)*np.ones((Nbatch), dtype="int32")
        previous = np.zeros((token1now.shape[0]), dtype="int32")
        for t in range(Tcap-1):
            temp[:, t] = wordidx
            if np.sum(count_eos) == Nbatch:
                break
            preds = m2.predict([imgnow, temp, keynow, token2nowhot*0, unpairnow],verbose=None)
            prednow = preds[0][:, t, :]
            if t > 1:
                for j in range(previous.shape[0]):
                    prednow[j,previous[j]] = -1e8
            wordidx = np.argmax(preds[0][:, t, :], axis=-1)
            previous = wordidx.copy()
            count_eos = np.logical_or(count_eos, wordidx == 2)
            pos_eos = np.minimum(pos_eos, count_eos*t+(1-count_eos)*Tcap)
        predcap = temp 
        for j in range(Nbatch):
            predcap[j,pos_eos[j]+2:] = 0
        output.append(predcap)
    output = np.concatenate(output, axis=0)
    return output
def greedy_batch_adjust_logit_soft(m2, imgs2b, token1capb, keytokenb, token2capb, masksb, Ntest2, Nbatch, usekey=True,earlystop=1e4):
    output = []
    outputsoft = []
    for idx in tqdm(range(min(Ntest2//Nbatch, earlystop))):
        i = idx*Nbatch
        imgnow = imgs2b[i:i+Nbatch]
        token1now = token1capb[i:i+Nbatch]
        token2now = token2capb[i:i+Nbatch]
        token2nowhot = to_categorical(token2now, num_classes=voc2, dtype='float32')
        keynow = keytokenb[i:i+Nbatch]
        if not usekey:
            keynow = 0*keynow
        unpairnow = np.ones((Nbatch, 1, 1), dtype="int32")
        temp = np.zeros(token1now.shape, dtype='int32')
        tempsoft = np.zeros_like(token2nowhot, dtype='float32')
        wordidx = 1  
        count_eos = np.zeros((Nbatch), dtype="int32")
        pos_eos = (Tcap-1)*np.ones((Nbatch), dtype="int32")
        previous = np.zeros((token1now.shape[0]), dtype="int32")
        for t in range(Tcap-1):
            temp[:, t] = wordidx
            if np.sum(count_eos) == Nbatch:
                break
            preds = m2.predict([imgnow, temp, keynow, token2nowhot*0, unpairnow],verbose=None)
            prednow = preds[0][:, t, :]
            if t > 1:
                for j in range(previous.shape[0]):
                    prednow[j,previous[j]] = min(prednow[j,previous[j]],1/voc2) 
            wordidx = np.argmax(prednow, axis=-1)
            tempsoft[:,t,:] =  prednow
            previous = wordidx.copy()
            count_eos = np.logical_or(count_eos, wordidx == 2)
            pos_eos = np.minimum(pos_eos, count_eos*t+(1-count_eos)*Tcap)
        predcap = temp 
        for j in range(Nbatch):
            predcap[j,pos_eos[j]+2:] = 0
        output.append(predcap)
        outputsoft.append(tempsoft)
    output = np.concatenate(output, axis=0)
    outputsoft = np.concatenate(outputsoft, axis=0)
    return output, outputsoft

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

def evaluate_batch(predcap, token2now, dictionaryAll, fprint=False):
    Ndata = predcap.shape[0]
    bleus = np.zeros((Ndata))
    rouges = np.zeros((Ndata))
    meteors = np.zeros((Ndata))
    for j in range(Ndata):
        gt = translate(j, dictionaryAll, token2now, 'targe', fprint=fprint)
        pred = translate(j, dictionaryAll, predcap, 'predi', fprint=fprint)
        if gt.strip()!="" and pred.strip()!="":
            bleus[j] = (calculate_bleu(pred, [gt])[0])
            rouges[j] = (calculate_rouge(pred, [gt]))
            meteors[j] = (calculate_meteor(pred, [gt]))
        else:
            print(f"{j}: gt={gt}; pred={pred}")
    """
    bleu = np.mean(np.array(bleus))
    rouge = np.mean(np.array(rouges))
    meteor = np.mean(np.array(meteors))
    print([bleu, rouge, meteor])
    """
    return bleus, rouges, meteors

def evaluate_batch_pred(predcap, token2now, dictionaryAll, fprint=False):
    output = predcap.copy()
    Ndata = predcap.shape[0]
    bleus = np.zeros((Ndata))
    rouges = np.zeros((Ndata))
    meteors = np.zeros((Ndata))
    for j in range(Ndata):
        gt = translate(j, dictionaryAll, token2now, 'targe', fprint=fprint)
        pred = translate(j, dictionaryAll, predcap, 'predi', fprint=fprint)
        if gt.strip()!="" and pred.strip()!="":
            bleus[j] = (calculate_bleu(pred, [gt])[0])
            rouges[j] = (calculate_rouge(pred, [gt]))
            meteors[j] = (calculate_meteor(pred, [gt]))
        else:
            print(f"{j}: gt={gt}; pred={pred}")
            output[j] = token2now[j]
    bleu = np.mean(np.array(bleus))
    rouge = np.mean(np.array(rouges))
    meteor = np.mean(np.array(meteors))
    print([bleu, rouge, meteor])
    return bleus, rouges, meteors, output

seed = 111
np.random.seed(seed)


"""
import csv
with open('eggs.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
Texts = np.loadtxt(open('newToken.csv',"rb"),delimiter=",",skiprows=0)

def loaddatascalar(filename, N):
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = []
        for idx in range(N):
            line = caption_data[idx]
            line = line.rstrip("\n")
            caption_mapping.append(int(line))

        return caption_mapping
"""

def loaddatascalar(filename, N):
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = []
        for idx in range(N):
            line = caption_data[idx]
            line = line.rstrip("\n")
            caption_mapping.append(int(line))

        return caption_mapping
    

def loaddata1bothtask(filename, foldername, N):
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = []
        for idx in range(N):
            line = caption_data[idx]
            line = line.rstrip("\n")
            """
            img_name, caption = line.split(",")
            img_name = img_name.strip()
            caption = caption.strip()
            """
            caption = line
            if len(foldername)==0:
                caption_mapping.append(int(caption))
            else:
                caption_mapping.append(foldername + caption)

        return caption_mapping


def loaddata1caption(filename, maxlength, N):
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = np.zeros((N,maxlength), dtype='int32')
        for idx in range(N):
            line = caption_data[idx]
            line = line.rstrip("\n")
            cells = line.split(",")
            wordarray = np.zeros((maxlength,)).astype(np.int32)
            for i in range(0, min(len(cells), maxlength)):
                if len(cells[i])>0: 
                    wordarray[i] = int(float(cells[i]))
                else:
                    break
            caption_mapping[idx,:] = wordarray

        return caption_mapping


def loaddata1key(filename, foldername, N):
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = []
        for idx in range(N):
            line = caption_data[idx]
            line = line.rstrip("\n")
            """
            img_name, caption = line.split(",")
            img_name = line.strip()
            """
            caption = line.strip()
            if len(foldername) == 0: 
                caption_mapping.append(int(caption))
            elif foldername == 'nofoldername': 
                caption_mapping.append(caption)
            else:
                caption_mapping.append(foldername + caption)

        return caption_mapping


def loaddata2key(filename, maxlength, N):
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = np.zeros((N,maxlength))
        for idx in range(N):
            line = caption_data[idx]
            line = line.rstrip("\n")
            cells = line.split(",")
            """
            img_name = cells[0].strip()
            """
            wordarray = np.zeros((maxlength,)).astype(np.int32)
            for i in range(0, min(len(cells), maxlength)):
                
                wordarray[i] = int(float(cells[i]))
            caption_mapping[idx,:] = wordarray

        return caption_mapping

def loaddata1insert(filename, foldername, N):
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = []
        for idx in range(N):
            line = caption_data[idx]
            line = line.rstrip("\n")
            caption_mapping.append(foldername + line)

        return caption_mapping

def loaddata2insert(filename, maxlength, N):
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = []
        for idx in range(N):
            line = caption_data[idx]
            line = line.rstrip("\n")
            cells = line.split(",")
            wordarray = np.zeros((maxlength,)).astype(np.int32)
            for i in range(0, min(len(cells), maxlength)):
                wordarray[i] = int(float(cells[i]))
            caption_mapping.append(wordarray)

        return caption_mapping

 
def loaddata2bothtask(filename, maxlength, N):
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = []
        for idx in range(N):
            line = caption_data[idx]
            line = line.rstrip("\n")
            cells = line.split(",")
            """
            img_name = cells[0].strip()
            wordarray = np.zeros((maxlength,)).astype(np.int32)
            for i in range(1, min(len(cells), maxlength)):
                wordarray[i-1] = int(cells[i])
            """
            wordarray = np.zeros((maxlength,)).astype(np.int32)
            for i in range(0, min(len(cells)-1, maxlength)):
                wordarray[i] = int(cells[i])
            caption_mapping.append(wordarray)

        return caption_mapping

def getGTtext(imgname):
    folder = 'resizedimg/'
    imgpathfull = loaddata1bothtask('imgpath.csv','resizedimg/',5491)
    tokendatafull = loaddata2bothtask('imgtoken.csv',70,5491)
    
    position = imgpathfull.index(imgname)
    text = np.expand_dims(tokendatafull[position], axis=0)
    return text


def evaluate_batch_log(predcap, token2now, dictionaryAll, fprint=False, Neval=1e4):
    output = predcap.copy()
    Ndata = predcap.shape[0]
    bleus = np.zeros((Ndata))
    rouges = np.zeros((Ndata))
    meteors = np.zeros((Ndata))
    log = ""
    for j in range(min(Ndata,Neval)):
        gt = translate(j, dictionaryAll, token2now, 'targe', fprint=fprint)
        pred = translate(j, dictionaryAll, predcap, 'predi', fprint=fprint)
        log = log+ f"{j} : gt={gt}; pred={pred} \n"
        if gt.strip()!="" and pred.strip()!="":
            bleus[j] = (calculate_bleu(pred, [gt])[0])
            rouges[j] = (calculate_rouge(pred, [gt]))
            meteors[j] = (calculate_meteor(pred, [gt]))
            log = log+ f"{j} : {bleus[j]} {rouges[j]} {meteors[j]} \n"
            if fprint:
                print(f"{j}: {bleus[j]} {rouges[j]} {meteors[j]} \n")
        """
        else:
            output[j, :-1] = token2now[j]
        """
    bleu = np.mean(np.array(bleus))
    rouge = np.mean(np.array(rouges))
    meteor = np.mean(np.array(meteors))
    print([bleu, rouge, meteor])
    print(output.shape)
    print(token2now.shape)
    return bleus, rouges, meteors, output, log
