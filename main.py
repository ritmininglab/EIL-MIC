from config import N, h1, w1, Nbatch, Tkey, Tcap, voc2, weightadj, dictappend, Nbatchkey
from utilInterpret import loaddata1key, loaddata2key
from imageio import imwrite
from utilIO import pretrain_caption_softlabel_softmax as re_train
from utilIO import get_test_caption_unpair as get_test3
from utilIO import get_test_caption_mask2_CrossEnt as get_test2
from utilInterpret import loaddata1caption, evaluate_batch_log
import pickle
from tqdm import tqdm
import csv
import numpy as np
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import cv2
import os
from utilIO import get_test_multilabelevidential as get_test
from utilpipe import get_cap, get_key, predkey
from utilIO import retrain_softmax_mixed_soft as retrain3soft
from utilpipe import logit2binary, evaluateUnc2, translateQuery, selectQuery, guiSelectKey
from utilpipe import updateKeywordPred, topKcandidate3update, topKcandidate3rank
from utilpipe import translatebeam1, guiSelectCap
from utilIO import re_train_keymask_mixed as re_train_key
from utilInterpret import greedy_batch_adjust_logit, evaluate_batch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)



dictionary = []
with open('TagDict.tsv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        dictionary.append(row[-2])
dictwordidx = {}
for j in range(len(dictionary)):
    word = dictionary[j]
    dictwordidx[word] = j

dictionaryAll = []
if dictappend is not None:
    dictionaryAll.append(dictappend)

with open('WordDict.tsv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        dictionaryAll.append(row[-2])

imgpath = loaddata1key('imgpath.csv', 'resizedimg/', N)
tokendata = loaddata2key('imgtag.csv', Tkey, N)
capdata = loaddata1caption('imgtoken.csv', Tcap, N)


mode = 1
if mode == 1:
    import random
    myorder = np.arange(N).tolist()
    random.Random(0).shuffle(myorder)
    imgpath3 = [imgpath[i] for i in myorder]
    tokendata3 = tokendata[myorder, ]
    capdata3 = capdata[myorder, ]

    bias = 0
    imgpath2 = imgpath3[bias:bias+N]
    tokendata2 = tokendata3[bias:bias+N]
    capdata2 = capdata3[bias:bias+N]



useevi = True




m0 = get_key()
m0.summary()
m2 = get_cap()
teacher = get_cap()
m2.summary()



useactive = False
usehard = True
if useactive:
    prefix = "mkeyact"
else:
    prefix = "mkeyseq"
if usehard:
    prefix = prefix+"h"
    

Npretrain = 3200
Nstepkey = Npretrain//Nbatchkey 

Nstep = 400 
Npretrain = Nstep
Npaired = None  



Nstep3 = 0
mygenerator2 = re_train(imgpath2, tokendata2, capdata2, Tkey, Tcap, Nbatch, Nstep, None, None, None,
                        Nstep3, voc2, weightadj=weightadj, unpairshift=Npaired, bias=0)

Nstep4 = 1000//Nbatch
mygenerator_val = re_train(imgpath2, tokendata2, capdata2, Tkey, Tcap, Nbatch, Nstep4, None, None, None,
                Nstep3, voc2, weightadj=weightadj, unpairshift=Npaired, bias=6400)


usekey = True
Ntest2 = 1000 
bias = 6400  
logging2interval=16
logging2heatup=16
imgs2b, token1capb, keytokenb, token2capb, masksb \
    = get_test2(imgpath2, tokendata2, capdata2, Tkey, Tcap, Ntest2, Nstep, bias, weightadj=weightadj)

imgpath3 = imgpath2[3200:6400]
tokendata3 = tokendata2[3200:6400]
capdata3 = capdata2[3200:6400]
imgs2c, token1capc, keytokenc, token2capc, masksc \
    = get_test2(imgpath2, tokendata2, capdata2, Tkey, Tcap, 3200, 0, 3200, weightadj=weightadj)
imgsk,token1k, token2k, targetk  = get_test(imgpath2,tokendata2,Tkey,h1,w1,Ntest2,0,bias,False)
token2khot = tf.keras.utils.to_categorical(token2k,2)



epochrecord = [39,59,79,99]

class CustomCallback(Callback):
    def __init__(self, logging2interval=10, logging2heatup=10, decay=0.8,teacher=None,step=""):
        super().__init__()
        self.history2 = {
            "bleu":[],
            "rouge":[],
            "meteor":[],
            }
        self.logging2interval = logging2interval
        self.logging2heatup = logging2heatup
        self.teacher = teacher
        self.decay = decay
        self.history3 = {
            "bleu":[],
            "rouge":[],
            "meteor":[],
            }
        self.step = step

    def on_epoch_end(self, epoch, logs=None):
        if self.teacher is not None:
            if epoch == self.logging2heatup:
                self.weightsma = self.model.get_weights()
                self.teacher.set_weights(self.weightsma)
            elif epoch > self.logging2heatup:
                weightsnew = self.model.get_weights()
                newma = []
                for i in range(len(self.weightsma)):
                    newlayer = self.weightsma[i]*self.decay + weightsnew[i]*(1-self.decay)
                    newma.append(newlayer)
                self.weightsma = newma
                self.teacher.set_weights(self.weightsma)
            m2 = self.teacher
            if epoch % self.logging2interval == 0 and epoch>=self.logging2heatup*2:
                predcap =  greedy_batch_adjust_logit(m2, imgs2b, token1capb, keytokenb, token2capb, masksb,
                                        Ntest2=Ntest2, Nbatch=8,usekey=usekey)
                bleus, rouges, meteors = evaluate_batch(predcap, token2capb, dictionaryAll, fprint=False)
                bleu = np.mean(np.array(bleus))
                rouge = np.mean(np.array(rouges))
                meteor = np.mean(np.array(meteors))
                print([bleu, rouge, meteor])
                self.history3["bleu"].append(bleu)
                self.history3["rouge"].append(rouge)
                self.history3["meteor"].append(meteor)
            if epoch in epochrecord:
                teacher.save_weights(f"{foldername}/{self.step}_{epoch}.h5")


foldername = "pipeUser"
if not os.path.exists(foldername):
   os.makedirs(foldername)





considerkey = True
considercap = True
Nselect = 16 
selected_byfar = set()
imgpath3 = imgpath2[3200:]
results_key = []
visualize = False
Nactive = 16 
Ntest = 1000
results = []
results2 = []
step_total = 5
accprior = 0.5
for step in tqdm(range(step_total)):
    if step==0: 
        if considerkey:
            Nstep3 = 0 
            mygenerator1 = re_train_key(imgpath2,tokendata2,Tkey,h1,w1,Nbatchkey,Nstep,None,None,None,False)
            history1 = m0.fit(mygenerator1,
                steps_per_epoch=Nstep+Nstep3,
                epochs=40)
        if considercap:
            callback2 = CustomCallback(teacher=teacher,logging2interval=logging2interval,
                                       logging2heatup=logging2heatup,step=step)
            history = m2.fit(mygenerator2,
                steps_per_epoch=Nstep,
                epochs=100,
                validation_data=mygenerator_val,
                validation_steps=Nstep)
    else:
        if considerkey:
            imgpath3,tokendata3,selected_byfar,mask3 = pickle.load( open( f"{foldername}/keydata_{prefix}_{step-1}.pkl", "rb" ) )
            mygenerator1 = re_train_key(imgpath2,tokendata2,Tkey,h1,w1,Nbatchkey,Nstep,imgpath3,tokendata3,mask3,False)
            m0 = get_key()
            history1 = m0.fit(mygenerator1,
                steps_per_epoch=Nstep,
                epochs=80)
        
        if considercap:
            capdata3a,capdata3soft = pickle.load( open( f"{foldername}/tokendata3a_{step-1}.pkl", "rb" ) )
            tokendata3a = np.argmax(tokendata3, axis=-1)
            mygenerator2 = retrain3soft(imgpath2,tokendata2,capdata2, Tkey,Tcap, Nbatch,Nstep,
                                    imgpath3,tokendata3a,capdata3a,capdata3soft,voc2,weightadj=weightadj)
            callback2 = CustomCallback(teacher=teacher,logging2interval=logging2interval,
                                       logging2heatup=logging2heatup,step=step)
            m2 = get_cap()
            history = m2.fit(mygenerator2,
                steps_per_epoch=Nstep,
                epochs=80, 
                validation_data=mygenerator_val,
                validation_steps=Nstep4,)

    teacher=m2
    
    if step<step_total-1:
        use_user_token_for_cap = True
        if useactive:
            bias = Npretrain
            Ntest = Npretrain
        else:
            bias = Npretrain +step*Nselect
            Ntest = Nselect 
        imgs,token1, token2, target  = get_test(imgpath2,tokendata2,Tkey,h1,w1,Ntest,Nstep,bias,False)
        unpairshift = None
        imgs2, token1cap, keytoken, token2cap, masks, unpair \
            = get_test3(imgpath2, tokendata2, capdata2, Tkey, Tcap, Ntest, 0, bias, weightadj=weightadj, unpairshift=unpairshift) 
        
        
        preds,vac,diss,ewc,prec,recall,f1 = predkey(m0, imgs,token1, token2, target, dictionary)
        
        evi = preds[0]
        S = np.sum(evi+1, axis=-1, keepdims=True)
        softlabelkey = (evi+1) / S
        
        vac, diss, weight = evaluateUnc2(evi, 0.8)
        predsbinary = logit2binary(preds[0])
        anno_binary = predsbinary.copy()
        anno_mask = np.zeros_like(predsbinary)
        predcap3 = np.zeros((token1.shape[0], token1.shape[1]+1), dtype="int32")
        for i in range(1):
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
            
            anno_binary[i:i+1] = updateKeywordPred(predsbinary[i:i+1], list1, ans2idx)
            for j in ans2idx:
                anno_mask[i,j] = 1
        

        
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
            token2now = tf.keras.utils.to_categorical(token2now,voc2)
            if use_user_token_for_cap:
                keynow = anno_binary[i:i+1]
            else:
                keynow = keytoken[i:i+1]
        
            candidates1[:, 0] = 1
            t = 0
            temp = candidates1[0:1]
            preds = m2.predict([imgnow,token1now,keynow,token2now,unpair[0:1]], verbose=None)
            lls = np.log(preds[0][0,t,:]+1e-8)
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
                    preds = m2.predict([imgnow,temp,keynow,token2now,unpair[0:1]], verbose=None)
                    lls = np.log(preds[0][0,t,:]+1e-8)
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
            
            ans3 = translatebeam1(i, dictionaryAll, candidates4[:cut], scores4[:cut], 
                                    printing=False)
            list2 = guiSelectCap(ans3)
            idx = list2[0] if len(list2)>0 else -1
            if idx>=0:
                predcap3[i:i+1,:-1] = candidates4[idx]
            else: 
                predcap3[i:i+1,:-1] = token1capc[i:i+1]
            
        
        ewc2 = accprior - accprior*np.squeeze(vac) - (accprior-0.5)*np.squeeze(diss)
        mask3now = ewc2*(1-anno_mask) + anno_mask
        if step>0:
            mask3 = np.concatenate([mask3, mask3now], axis=0)
        else:
            mask3 = mask3now
        
        gt1hot = tf.keras.utils.to_categorical(anno_binary, 2)
        temp = anno_mask[:,:,np.newaxis]
        softlabelkey = softlabelkey*(1-temp) + gt1hot*temp
        if usehard:
            softlabelkey = np.argmax(softlabelkey, axis=-1)
            softlabelkey = tf.keras.utils.to_categorical(softlabelkey, 2)
        pickle.dump( [imgpath3,softlabelkey,selected_byfar,mask3], open( f"{foldername}/keydata_{prefix}_{step}.pkl", "wb" ) )
        
        predcapsoft = tf.keras.utils.to_categorical(predcap3[:,1:], voc2)
        if step>0:
            predcap3 = np.concatenate([capdata3a, predcap3], axis=0)
            predcapsoft = np.concatenate([capdata3soft, predcapsoft], axis=0)
        pickle.dump([predcap3,predcapsoft], open( f"{foldername}/tokendata3a_{step}.pkl", "wb" ) )

    if considerkey:
        preds,vac,diss,ewc,prec,recall,f1 = predkey(m0, imgsk,token1k, token2k, targetk, dictionary)
        results_key.append([prec, recall, f1])
        
        predsbinary = preds[0][:,:,1] > preds[0][:,:,0]
        accprior = np.mean(predsbinary==token2k)

    
    if considercap:
        predcap =  greedy_batch_adjust_logit(teacher, imgs2b, token1capb, predsbinary, token2capb, masksb,
                                Ntest2=1000, Nbatch=8,usekey=usekey)
        bleus, rouges, meteors, output, log = evaluate_batch_log(predcap, \
                               token2capb, dictionaryAll, fprint=False, Neval=predcap.shape[0])
        bleu = np.mean(np.array(bleus))
        rouge = np.mean(np.array(rouges))
        meteor = np.mean(np.array(meteors))
        results.append([bleu, rouge, meteor])