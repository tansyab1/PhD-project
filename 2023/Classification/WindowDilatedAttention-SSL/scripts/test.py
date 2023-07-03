from concurrent.futures import process
import glob
# import csv
import os

import cv2
import numpy as np
# from functools import reduce
from tqdm import tqdm
# from mask import create_mask

mask_dir = "src/Data/mask/"
process_mask = cv2.imread("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/distorted_images/mask.png")

def applyui(image, mask):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.where(hsv[:,:,2]>20, mask/255*hsv[:,:,2],hsv[:,:,2])
    res = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return np.array(res,dtype=np.uint8)

img = cv2.imread("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/resultsforPaper/ui/proposed/bca26705313a4644_18164.PNG")
# convert to hsv
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

hsv[:,:,2] = hsv[:,:,2]*0.8
# hsv[:,:,2] = np.where(hsv[:,:,2]>255, 255, hsv[:,:,2])
# cv2.imshow('image',hsv[:,:,2])
# wait = cv2.waitKey(0)
# convert back to bgr
bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

final = np.where(process_mask <10, process_mask, img)
cv2.imshow('image',final)
wait = cv2.waitKey(0)
# convert to 0 - 255
final = np.array(final,dtype=np.uint8)

# save image
cv2.imwrite('/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/resultsforPaper/ui/ref/250_112.png',final)