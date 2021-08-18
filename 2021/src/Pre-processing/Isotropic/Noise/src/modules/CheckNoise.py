import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os


def estimateStandardDeviation(image):
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
    width, height = image.shape
    operator = laplaceElement()
    return np.sqrt(math.pi / 2) * 1 / (6 * (width-2) * (height-2)) * np.sum(np.abs(cv2.filter2D(image, -1, operator)))


def laplaceElement():
    """
    Create a Laplace filter element.

    Returns
    -------
    ndarray
        Laplace filter element.

    """
    L1 = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]], dtype=np.float)
    L2 = np.array([[1, 0, 1],
                   [0, -4, 0],
                   [1, 0, 1]], dtype=np.float)
    return L2 - 2*L1


def check_noise(file_path, thresh=0.1):
    """
    Check if the file is noise or not.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    bool
        True if the file is noise, False if not.

    """
    img = cv2.imread(file_path, 0)
    sigma_n = estimateStandardDeviation(img)
    if sigma_n < thresh:
        return False
    else:
        return True


def readImagefromFolder(folder_path, file_type):
    """read image from folder"""
    file_list = []
    sigma_list = []
    for file in os.listdir(folder_path):
        if file.endswith(file_type):
            file_list.append(file)
    for i in range(len(file_list)):
        file_list[i] = folder_path + file_list[i]
        img = cv2.imread(file_list[i], 0)
        sigma_n = estimateStandardDeviation(img)
        sigma_list[i] = sigma_n


def plotHistogram(arr):
    """
    Plot the histogram of the array.

    Parameters
    ----------
    arr : ndarray
        Array to plot the histogram.

    """
    plt.hist(arr.ravel(), 256, [0, 256])
    plt.show()


if __name__ == "__main__":
    img_list = readImagefromFolder('./data/', '.jpg')
    plotHistogram(img_list)
