from concurrent.futures import process
import glob
# import csv
import os

import cv2
import numpy as np
# from functools import reduce
from tqdm import tqdm
from mask import create_mask

mask_dir = "src/Data/mask/"
process_mask = cv2.imread("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos/mask.png")

def applyui(image, mask):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.where(hsv[:,:,2]>20, mask/255*hsv[:,:,2],hsv[:,:,2])
    res = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return np.array(res,dtype=np.uint8)

sizes = [250,150,200,100]
angles = [112,168,224]
datapath= "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos/ref/*.mp4"

ui_save_folder = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos/UI/'
for size in tqdm(sizes):
    for angle in angles:
        mask=cv2.imread(mask_dir+str(size)+"_"+str(angle)+".png",cv2.IMREAD_GRAYSCALE)
        if not os.path.exists(ui_save_folder + str(size) + '/' + str(angle) + '/'):
            os.makedirs(ui_save_folder + str(size) + '/' + str(angle) + '/')
        for file in tqdm(glob.glob(datapath)):
            output_video = cv2.VideoWriter(
                ui_save_folder + str(size) + '/'+ str(angle) + '/' + os.path.basename(file), cv2.VideoWriter_fourcc(*'mp4v'), 30, (336, 336))
            cap = cv2.VideoCapture(file)
            # Check if camera opened successfully
            if (cap.isOpened() is False):
                print("Error opening video stream or file")

            # Read until video is completed
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret is True:
                    noise_img = applyui(frame, mask=mask)
                    finalnoise= np.where(process_mask <10, frame, noise_img)
                    # Display the resulting frame
                    output_video.write(finalnoise)

                # Break the loop
                else:
                    cap.release()
                    break
print("UI- Done")