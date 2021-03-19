# -*- coding: utf-8 -*-
import numpy as np
import pickle
from PIL import Image
from scipy.signal import convolve2d
import os.path

pickledPsfFilename =os.path.join(os.path.dirname( __file__),"psf.pkl")

with open(pickledPsfFilename, 'rb') as pklfile:
    psfDictionary = pickle.load(pklfile)


def PsfBlur(img, psfid):
    imgarray = np.array(img, dtype="float32")
    kernel = psfDictionary[psfid]
    if imgarray.ndim==3 and imgarray.shape[-1]==3:
        convolved = np.stack([convolve2d(imgarray[...,channel_id], 
                    kernel, mode='same', 
                    fillvalue=255.0).astype("uint8") 
                    for channel_id in range(3)], axis=2)
    else:
        convolved = convolve2d(imgarray, kernel, mode='same', fillvalue=255.0).astype("uint8")
    img = Image.fromarray(convolved)
    return img
    
def PsfBlur_random(img):
    psfid = np.random.randint(0, len(psfDictionary))
    return PsfBlur(img, psfid)
    
    
