
import numpy as np
import cv2
# import csv
import os
import glob
from tqdm import tqdm
# from functools import reduce
from skimage.util import random_noise


def create_noise(image, sigma, mean=0):
    noise_img = random_noise(image, mode='gaussian',
                             mean=mean,  var=(sigma/255)**2, clip=True)
    return np.array(255*noise_img, dtype='uint8')


save_folder = 'src/Data/labelled/outNoise/level5/'

for file in tqdm(glob.glob("src/Data/labelled/ref/*.mp4")):
    output_video = cv2.VideoWriter(
        save_folder + os.path.basename(file), cv2.VideoWriter_fourcc(*'mp4v'), 30, (336, 336))
    cap = cv2.VideoCapture(file)
    # Check if camera opened successfully
    if (cap.isOpened() is False):
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is True:
            noise_img = create_noise(frame, sigma=30)
            # Display the resulting frame
            output_video.write(noise_img)

        # Break the loop
        else:
            cap.release()
            break
