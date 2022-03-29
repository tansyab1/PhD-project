import glob
# import csv
import os

import cv2
import numpy as np
# from functools import reduce
from skimage.util import random_noise
from tqdm import tqdm


def create_noise(image, sigma, mean=0):
    noise_img = random_noise(image, mode='gaussian',
                             mean=mean, var=(sigma / 255) ** 2, clip=True)
    return np.array(255 * noise_img, dtype='uint8')


sigmas = [5, 10, 15, 20, 30, 40]
save_folder = 'D:/Datasets/kvasircapsule/labelled_videos/Noise/'
for sigma in sigmas:
    for file in tqdm(glob.glob("D:/Datasets/kvasircapsule/labelled_videos/ref/*.mp4")):
        output_video = cv2.VideoWriter(
            save_folder + str(sigma) + '/' + os.path.basename(file), cv2.VideoWriter_fourcc(*'mp4v'), 30, (336, 336))
        cap = cv2.VideoCapture(file)
        # Check if camera opened successfully
        if (cap.isOpened() is False):
            print("Error opening video stream or file")

        # Read until video is completed
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret is True:
                noise_img = create_noise(frame, sigma=sigma)
                # Display the resulting frame
                output_video.write(noise_img)

            # Break the loop
            else:
                cap.release()
                break
