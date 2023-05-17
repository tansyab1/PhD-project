import numpy as np
import cv2
# import csv
import os
import glob
from tqdm import tqdm
from functools import reduce
from skimage.util import random_noise

# check version opencv
print(cv2.__version__)


def create_noise(image, sigma, mean=0):
    noise_img = random_noise(image, mode='gaussian',
                             mean=mean,  var=(sigma/255)**2, clip=True)
    return np.array(255*noise_img, dtype='uint8')


def addNoise():

    process_mask = cv2.imread(
        "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/mask.png")
    sigmas = [5, 10, 15, 30]

    datapath = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/main/final/*.mp4"
    noise_save_folder = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/main/Noise/'
    # ref_folder = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest/videoReadGUI/ref-fps5/'

    for sigma in tqdm(sigmas):
        if not os.path.exists(noise_save_folder + str(sigma) + '/'):
            os.makedirs(noise_save_folder + str(sigma) + '/')
        for file in tqdm(glob.glob(datapath)):
            # orifinal video writer for the reference

            # output writer for the noise video
            output_video = cv2.VideoWriter(
                noise_save_folder + str(sigma) + '/' + os.path.basename(file), cv2.VideoWriter_fourcc(*'avc1'), 30, (336, 336))

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
    
if __name__ == "__main__":
    saved_dir = "/Users/sy/Downloads/ref_videos/fps5/"
    for video in tqdm(glob.glob("/Users/sy/Downloads/ref_videos/*.mp4")):
        # video = "2021/src/Data/labelled_video/setFPS.mp4"
        # get video name
        
        video_name = os.path.basename(video)
        
        # orifinal video writer for the reference
        ref_video = cv2.VideoWriter(saved_dir + video_name, cv2.VideoWriter_fourcc(*'avc1'), 5, (336, 336))

        cap = cv2.VideoCapture(video)
        ret, frame = cap.read()
        # Check if camera opened successfully
        if (cap.isOpened() is False):
            print("Error opening video stream or file")

        while ret:  # Use the ret to determin end of video
            ref_video.write(frame) # Write frame
            ret, frame = cap.read()
