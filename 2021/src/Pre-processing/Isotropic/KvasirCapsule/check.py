
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

save_folder ='/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/KvasirCapsule/coefs_unlabelled/'

for file in tqdm(glob.glob("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/unlabelled_videos/*.mp4")):
    count = 0
    cap = cv2.VideoCapture(file)
    # Check if camera opened successfully
    if (cap.isOpened() is False):
        print("Error opening video stream or file")

    ui = []
    noise = []
    blur = []
    # Read until video is completed
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is True:
            # Display the resulting frame
            ui_coefficient = getIHED(frame)
            noise_coefficient = estimateStandardDeviation(frame)
            blur_coefficient = getVarianofLaplace(frame)

            count += 30 # i.e. at 30 fps, this advances 1 second
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            ui.append(ui_coefficient)
            noise.append(noise_coefficient)
            blur.append(blur_coefficient)

        # Break the loop
        else:
            cap.release()
            break

    with open(save_folder + os.path.basename(file)+'.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(ui, noise, blur))
