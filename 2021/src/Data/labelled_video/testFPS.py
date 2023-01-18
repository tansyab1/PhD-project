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

def testFPS():

    # process_mask = cv2.imread("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/mask.png")
    # sigmas = [5,10,15, 30]

    datapath= "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest/videoReadGUI/select20/ref_3c8d5f0b90d7475d.mp4"
    testFPS_folder = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest/testFPS/'

    FPS = [5, 10, 15, 20, 25, 30]
    for fps in tqdm(FPS):
        for file in tqdm(glob.glob(datapath)):
            # output writer for the noise video
            output_video = cv2.VideoWriter(
                testFPS_folder + str(fps) + '/' + os.path.basename(file), cv2.VideoWriter_fourcc(*'avc'), fps, (336, 336))

            cap = cv2.VideoCapture(file)
            # Check if camera opened successfully
            if (cap.isOpened() is False):
                print("Error opening video stream or file")

            # Read until video is completed
            while (cap.isOpened()):
                ret, frame = cap.read()
                # ref_video.write(frame)
                if ret is True:
                    final = frame.copy()
                    # Display the resulting frame
                    output_video.write(final)

                # Break the loop
                else:
                    cap.release()
                    # output_video.release()
                    break

                
            
                
    # notify when the process is done
    print("test FPS Done")

if __name__ == "__main__":
    testFPS()

