import cv2
import numpy as np


def create_mask():
    datapath = "D:/Datasets/kvasircapsule/labelled_videos/test.jpg"
    frame = cv2.imread(datapath)
    out = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = np.where(out < 20, 0, 255)
    return output
