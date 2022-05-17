import glob
# import csv
import os

import cv2
import numpy as np
# from functools import reduce
from tqdm import tqdm
from mask import create_mask

mask = cv2.imread("D:/Datasets/kvasircapsule/labelled_videos/mask.png")

def apply_motion_blur(image, size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    # k[(size -1) // 2, :] = np.ones(size, dtype=np.float32)
# cach 1
    if round(np.cos(angle)):
      k[(size//2),(size//2)]=1
      for i in range(size):
        for j in range(size):
          if (j-size//2):
            # print((i-size//2)/(j-size//2))
            # print(int(-np.tan(np.pi)))
            if np.sqrt((i-size//2)**2+(j-size//2)**2)<=size and (i-size//2)/(j-size//2) == -round(np.tan(angle)):
              k[i,j]=1
    else:
      k[:,(size -1) // 2] = np.ones(size, dtype=np.float32)
# cach 2
    # k = cv2.warpAffine(k, cv2.getRotationMatrix2D(
    #     (size / 2 - 0.5, size / 2 - 0.5), angle, 1.0), (size, size))
    # k = np.where(k<np.max(k), 0,255)
    # k = k * (1.0 / np.sum(k))
    dst = cv2.filter2D(image, -1, k)
    out = np.where(mask == np.array([0, 0, 0]), image, dst)
    return out


def apply_defocus_blur(image, sigma):
    dst = cv2.GaussianBlur(image, (int(6 * sigma) + 1, int(6 * sigma) + 1), sigma)
    out = np.where(mask == np.array([0, 0, 0]), image, dst)
    return out


sigmas = [0.75,1, 2, 3, 5]
angles = [0, 45, 90, 135]
sizes = [5, 10, 15, 25]

datapath= "D:/Datasets/kvasircapsule/labelled_videos/ref/*.mp4"
# defocus_save_folder = 'D:/Datasets/kvasircapsule/labelled_videos/Blur/Defocus/'
# for sigma in tqdm(sigmas):
#     if not os.path.exists(defocus_save_folder + str(sigma) + '/'):
#         os.makedirs(defocus_save_folder + str(sigma) + '/')
#     for file in tqdm(glob.glob(datapath)):
#         output_video = cv2.VideoWriter(
#             defocus_save_folder + str(sigma) + '/' + os.path.basename(file), cv2.VideoWriter_fourcc(*'mp4v'), 30, (336, 336))
#         cap = cv2.VideoCapture(file)
#         # Check if camera opened successfully
#         if (cap.isOpened() is False):
#             print("Error opening video stream or file")
#
#         # Read until video is completed
#         while (cap.isOpened()):
#             ret, frame = cap.read()
#             if ret is True:
#                 noise_img = apply_defocus_blur(frame, sigma=sigma)
#                 finalnoise = np.where(mask < 10, frame, noise_img)
#                 # Display the resulting frame
#                 output_video.write(finalnoise)
#
#             # Break the loop
#             else:
#                 cap.release()
#                 break
#
# print("Defocus Blur- Done")

motion_save_folder = 'D:/Datasets/kvasircapsule/labelled_videos/Blur/Motion/'
for size in tqdm(sizes):
    for angle in angles:
        if not os.path.exists(motion_save_folder + str(size) + '/' + str(angle) + '/'):
            os.makedirs(motion_save_folder + str(size) + '/' + str(angle) + '/')
        for file in tqdm(glob.glob(datapath)):
            output_video = cv2.VideoWriter(
                motion_save_folder + str(size) + '/'+ str(angle) + '/' + os.path.basename(file), cv2.VideoWriter_fourcc(*'mp4v'), 30, (336, 336))
            cap = cv2.VideoCapture(file)
            # Check if camera opened successfully
            if (cap.isOpened() is False):
                print("Error opening video stream or file")

            # Read until video is completed
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret is True:
                    noise_img = apply_motion_blur(frame, size=size, angle=angle)
                    finalnoise= np.where(mask <10, frame, noise_img)
                    # Display the resulting frame
                    output_video.write(finalnoise)

                # Break the loop
                else:
                    cap.release()
                    break
print("Motion Blur- Done")