
import numpy as np
import cv2
import glob
import os
import csv
from tqdm import tqdm
from functools import reduce


def localneighbors(im, i, j, d=1):
    b = im[i-d:i+d+1, j-d:j+d+1].flatten()
    # remove the element (i,j)
    return np.mean(b), Average(np.hstack((b[:len(b)//2], b[len(b)//2+1:])))

    """get the neighbors of the pixel (i,j)"""


def estimateuneven(img, d=1, name=None):
    """
    Estimate the standard deviation of the image.

    Parameters
    ----------
    image : ndarray
        Image to estimate the standard deviation.

    Returns
    -------
    float
        The standard deviation of the image.

    """

    illmask = cv2.medianBlur(img, 201)
    BsMat = np.zeros(img.shape)
    res = np.zeros(img.shape)
    BgMat = np.zeros(img.shape)
    image = cv2.copyMakeBorder(illmask, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    image_global = cv2.copyMakeBorder(illmask, d, d, d, d, cv2.BORDER_REFLECT)
    width, height = image.shape
    width_global, height_global = image_global.shape

    for i in range(d, width_global-d):
        for j in range(d, height_global-d):
            if image_global[i, j]:
                Bg, _ = localneighbors(image_global, i, j, d=3)
                BgMat[i-d, j-d] = Bg
    for i in range(1, width-1):
        for j in range(1, height-1):
            if image[i, j]:
                _, Bs = localneighbors(image, i, j, d=1)
                BsMat[i-1, j-1] = Bs

    Ba = 0.923*BsMat + 0.077*np.mean(illmask)
    res = np.abs(Ba-illmask)/Ba
    cv2.imwrite("test"+name, res)
    return np.mean(res)


def Average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)


if __name__ == '__main__':
    #     img = np.ones((7, 7))
    #     a, b = localneighbors(img, 0, 0, d=1)
    #     print(b)

    stdss = []
    names = []

    for file in tqdm(glob.glob("/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/src/img_ppt/agic/video*.png")):
        print(file)
        names.append(os.path.basename(file))
        img = cv2.imread(file, 0)
        std = estimateuneven(img, d=3, name=os.path.basename(file))
        stdss.append(std)

    # with open('test_artificial.csv', 'w') as f:
    #     writer = csv.writer(f, delimiter='\t')
    #     writer.writerows(zip(names, stdss))
