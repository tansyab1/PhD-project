import numpy as np
import cv2

def getIHED(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    median = cv2.medianBlur(img[:, :, 2], 201)
    ori = median.copy()
    equ = cv2.equalizeHist(median)
    res = np.zeros(equ.shape)
    for i in range(0, ori.shape[0]):
        for j in range(0, ori.shape[1]):
            res[i][j] = np.maximum(ori[i][j], equ[i][j])-np.minimum(ori[i][j], equ[i][j])

    mean, std = cv2.meanStdDev(res)
    dot = np.mean(ori)
    return std/dot