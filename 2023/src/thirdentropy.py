# implement the third entropy of the image using coocurrence matrix

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, greycoprops
from skimage import data


# read the image
img1 = cv2.imread('2023/src/lena.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('2023/src/lena.bmp', cv2.IMREAD_GRAYSCALE)

def thirdEntropy(img1, img2):
    # convert the image to gray scale
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gCoMatfinal = np.zeros((256, 256, 256))

    # calculate the 3D coocurrence matrix in 4 directions and inter frame distance 1
    # 0 degree
    gCoMat = graycomatrix(img1, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True)
    gCoMatspatial = (gCoMat[:, :, 0, 0] + gCoMat[:, :, 0, 1] + gCoMat[:, :, 0, 2] + gCoMat[:, :, 0, 3])
    
    normalized_matrix = gCoMatspatial / np.sum(gCoMatspatial)
    
     # extend the 3D coocurrence matrix to 5D
    img1_flat = np.reshape(img1, (img1.shape[0] * img1.shape[1], 1))
    img2_flat = np.reshape(img2, (img2.shape[0] * img2.shape[1], 1))
    
    img_flat = np.concatenate((img1_flat, img2_flat), axis=1)
    
    gCoMat_temp = graycomatrix(img_flat, [1], [0], levels=256, symmetric=True)
    normalized_matrix_temp = gCoMat_temp[:, :, 0, 0] / np.sum(gCoMat_temp[:, :, 0, 0])
    
    
    
    # gCoMat final is the element-wise product of gCoMat and gCoMat_temp
    thirdEntropy = 0
    
    gCoMatfinal = normalized_matrix[:, :, np.newaxis] * normalized_matrix_temp[:, np.newaxis, :]
    
    # gCoMatfinal = gCoMatspatial_reshaped * normalized_matrix_temp
    # print("here")
    nonzero_indices = np.nonzero(gCoMatfinal)
    thirdEntropy = -np.sum(gCoMatfinal[nonzero_indices] * np.log2(gCoMatfinal[nonzero_indices]))
    
                    
    return thirdEntropy

