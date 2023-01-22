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


def addNoise():

    process_mask = cv2.imread(
        "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/mask.png")
    sigmas = [5, 10, 15, 30]

    datapath = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest/videoReadGUI/select20/*.mp4"
    noise_save_folder = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest/videoReadGUI/fps5/Noise/'
    ref_folder = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest/videoReadGUI/ref-fps5/'

    for file in tqdm(glob.glob(datapath)):
        # orifinal video writer for the reference
        ref_video = cv2.VideoWriter(
            ref_folder + os.path.basename(file), cv2.VideoWriter_fourcc(*'avc1'), 5, (336, 336))

        cap = cv2.VideoCapture(file)
        # Check if camera opened successfully
        if (cap.isOpened() is False):
            print("Error opening video stream or file")

        # Read until video is completed
        while (cap.isOpened()):
            ret, frame = cap.read()
            ref_video.write(frame)
            # Break the loop
        else:
            cap.release()
            ref_video.release()
            break

    for sigma in tqdm(sigmas):
        if not os.path.exists(noise_save_folder + str(sigma) + '/'):
            os.makedirs(noise_save_folder + str(sigma) + '/')
        for file in tqdm(glob.glob(datapath)):
            # orifinal video writer for the reference

            # output writer for the noise video
            output_video = cv2.VideoWriter(
                noise_save_folder + str(sigma) + '/' + os.path.basename(file), cv2.VideoWriter_fourcc(*'avc1'), 5, (336, 336))

            cap = cv2.VideoCapture(file)
            # Check if camera opened successfully
            if (cap.isOpened() is False):
                print("Error opening video stream or file")

            # Read until video is completed
            while (cap.isOpened()):
                ret, frame = cap.read()
                # ref_video.write(frame)
                if ret is True:
                    noise_img = create_noise(frame, sigma=sigma)
                    finalnoise = np.where(process_mask < 10, frame, noise_img)
                    # Display the resulting frame
                    output_video.write(finalnoise)

                # Break the loop
                else:
                    cap.release()
                    # output_video.release()
                    break

    # notify when the process is done
    print("Noise Done")
