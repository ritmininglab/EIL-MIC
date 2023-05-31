import os


import scipy.io
import scipy.misc
import numpy as np
from numpy import expand_dims


from skimage.transform import resize

from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.patches import Rectangle


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
