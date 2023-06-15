# -*- coding: utf-8 -*-
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
            median = cv2.medianBlur(img[:, :, 2], img[:, :, 2].shape[0]//4)
            ori = median.copy()
            equ = cv2.equalizeHist(median)
            res = np.zeros(equ.shape)
            # equ = clahe.apply(median)
            for i in range(0, ori.shape[0]):
                for j in range(0, ori.shape[1]):
                    res[i][j] = np.maximum(ori[i][j], equ[i][j])-np.minimum(ori[i][j], equ[i][j])

            mean, std = cv2.meanStdDev(res)
            dot = np.mean(ori)
            stds.append(std/dot)
            # Press Q on keyboard to  exit
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break

        # Break the loop
        else:
            break
    stdss.append(Average(stds))

with open('./ihed_test.csv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(names, stdss))
