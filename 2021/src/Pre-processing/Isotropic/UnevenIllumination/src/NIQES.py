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
from niqe_module import niqe
# import matplotlib.pyplot as plt
# from skimage.util import random_noise
import glob
# import seaborn as sns
from tqdm import tqdm
# import matplotlib as mpl
# import pywt
from functools import reduce
from scipy.stats import kurtosis
# from google.colab.patches import cv2_imshow
stdss = []
names = []


def Average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
for file in tqdm(glob.glob("/home/nguyentansy/DATA/nguyentansy/PhD-work/Datasets/LVQ/uneven_illum/video*.avi")):
    cap = cv2.VideoCapture(file)
    names.append(os.path.basename(file))
    # Check if camera opened successfully
    if (cap.isOpened() is False):
        print("Error opening video stream or file")

    niqe_ks = []
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret is True:
            # Display the resulting frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = img/ 255
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.log(np.abs(fshift))

            # calculate the kurtosis and standard deviation of the image
            kurt = kurtosis(magnitude_spectrum, None, fisher=False)
            mean, std = cv2.meanStdDev(magnitude_spectrum, mask=None)

            # calculate the niqe
            niqe_img = niqe(img)
            niqe_k = kurt/std * niqe_img

            # calculate the NIQE score

            # dot = np.mean(ori)
            niqe_ks.append(niqe_k)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break
    stdss.append(Average(niqe_ks).item())

with open('/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/res/niqes.csv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(names, stdss))
