
import numpy as np
import cv2
import csv
import os
import glob
from tqdm import tqdm
from functools import reduce
from checkNoise import estimateStandardDeviation
from checkBlur import getVarianofLaplace
from checkUI import getIHED

name = 'Ampulla of vater'
save_folder ='D:/Datasets/kvasircapsule/labelled_images/'

for file in tqdm(glob.glob("D:/Datasets/kvasircapsule/labelled_images/"+name+"/*.mp4")):
    ui = []
    noise = []
    blur = []
    frame =cv2.imread(file)
    # Display the resulting frame
    ui_coefficient = getIHED(frame)
    noise_coefficient = estimateStandardDeviation(frame)
    blur_coefficient = getVarianofLaplace(frame)
    
    ui.append(ui_coefficient)
    noise.append(noise_coefficient)
    blur.append(blur_coefficient)


with open(save_folder + name+'.csv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(ui, noise, blur))
