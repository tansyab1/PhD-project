# -*- coding: utf-8 -*-
"""IHED.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KqU2WLfGKmhYbMu6j4X4gU56LZK2aM4e
"""

# ! mkdir /content/hyperkvasir/
# #Download the dataset
# ! unzip '/content/drive/MyDrive/Colab Notebooks/PhD-project/Dataset/hyperkvasir.zip' -d "/content/hyperkvasir/"


# ! mkdir /content/kvasir_capsule/
# #Download the dataset
# ! unzip '/content/drive/MyDrive/Colab Notebooks/PhD-project/Dataset/kvasircapsule.zip' -d "/content/kvasir_capsule/"

import numpy as np
import cv2
# import math
import csv
import os
# import matplotlib.pyplot as plt
# from skimage.util import random_noise
import glob
from jnc import jnc
# import seaborn as sns
from tqdm import tqdm
# import matplotlib as mpl
# import pywt
from functools import reduce
# from google.colab.patches import cv2_imshow
stdss = []
names = []


def Average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
clahe = cv2.createCLAHE(tileGridSize=(8, 8), clipLimit=2.0)
for file in tqdm(glob.glob("/home/nguyentansy/DATA/nguyentansy/PhD-work/Datasets/LVQ/uneven_illum/video*.avi")):
    cap = cv2.VideoCapture(file)
    names.append(os.path.basename(file))
    # Check if camera opened successfully
    if (cap.isOpened() is False):
        print("Error opening video stream or file")

    stds = []
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret is True:
            # Display the resulting frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            jnc_level = jnc(img[:, :, 2], d=1, name=os.path.basename(file))

            stds.append(jnc_level)

        # Break the loop
        else:
            break
    stdss.append(Average(stds))

with open('src/Pre-processing/Isotropic/UnevenIllumination/res/jnc.csv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(names, stdss))