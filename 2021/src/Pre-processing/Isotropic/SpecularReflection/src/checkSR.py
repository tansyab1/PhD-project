import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def calnumofwhitepixel(img):
    """
    Calculate the number of white pixel in the image
    :param img: image
    :return: number of white pixel
    """
    return img.shape[0] * img.shape[1] - np.count_nonzero(img)


def readImageListfromFolder(folder):
    """
    Read all images in the folder
    :param folder: folder path
    :return: list of image
    """
    img_list = []
    for file in os.listdir(folder):
        img_list.append(cv2.imread(os.path.join(folder, file)))
    return img_list


def plotHistogram(arr):
    """
    plot histogram of the number of white pixel
    :param arr: array of number of white pixel
    :return:
    """
    plt.hist(arr, bins=256)
    plt.show()


def checkSR(folder_path):
    """ check SR for all images in the folder"""
    img_list = readImageListfromFolder(folder_path)
    num_white_list = []
    for img in img_list:
        num_white = calnumofwhitepixel(img)
        num_white_list.append(num_white)

    plotHistogram(num_white_list)
    return num_white_list


if __name__ == "_main_":
    checkSR('../../../samples/data/')
