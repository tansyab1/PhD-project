import glob
# import csv
import os

import cv2
import numpy as np
# from functools import reduce
from tqdm import tqdm
from mask import create_mask


# find the centroid of the image
def find_centroid(img):
    # find the contours of the image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # find the centroid of the contours
    M = cv2.moments(img)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy

# find the direction of the image
def find_direction(x1, y1, x2, y2):
    # find the slope of the line
    if (x2 - x1) == 0:
        return 90
    # find the angle of the line
    slope = (y2 - y1) / (x2 - x1)
    angle = np.arctan(slope)
    degree = np.degrees(angle) if np.degrees(angle) > 0 else np.degrees(angle) + 180
    return degree


# find the motion direction of three images via background subtraction
def calDirection(set_imgs):
    # convert the images to gray scale
    gray_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in set_imgs]
    # subtract the first image with the second image in the set
    diff_images_1 = cv2.absdiff(gray_imgs[0], gray_imgs[1])
    # subtract the second image with the third image in the set
    diff_images_2 = cv2.absdiff(gray_imgs[1], gray_imgs[2])
    # convert the images to binary
    ret, binary_images_1 = cv2.threshold(diff_images_1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, binary_images_2 = cv2.threshold(diff_images_2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # find the centroid of each biary image
    x1, y1 = find_centroid(binary_images_1)
    x2, y2 = find_centroid(binary_images_2)
    # find the motion direction of the two images
    direction = find_direction(x1, y1, x2, y2)
    return direction
