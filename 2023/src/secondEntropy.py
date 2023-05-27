# implement the second entropy of the image using coocurrence matrix

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix
from skimage import data

# read the image
img = cv2.imread('2023/src/lena.bmp', cv2.IMREAD_GRAYSCALE)

# calculate the coocurrence matrix in 4 directions
# 0 degree
def secondEntropy(img):
    gCoMat = graycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True)

    # calculate the total coocurrence matrix
    gCoMatTotal = gCoMat[:, :, 0, 0] + gCoMat[:, :, 0, 1] + gCoMat[:, :, 0, 2] + gCoMat[:, :, 0, 3]
    # sum the elements in the row of the total coocurrence matrix
    # print(gCoMatTotal)
    normalized_matrix = gCoMatTotal / np.sum(gCoMatTotal)

    # Calculate marginal probabilities
    # row_sums = np.sum(normalized_matrix, axis=1)
    # col_sums = np.sum(normalized_matrix, axis=0)
    secondEntropy = 0
    for i in range(256):
        for j in range(256):
            if gCoMatTotal[i, j] > 0:
                secondEntropy += -normalized_matrix[i, j] * math.log2(normalized_matrix[i, j])
    

    return secondEntropy
            
