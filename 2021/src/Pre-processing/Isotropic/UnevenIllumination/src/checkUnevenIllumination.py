import math
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os


def createGaussianFilter(size, sigma):
    """create the gaussian filter
    size: the size of the filter
    sigma: the standard deviation of the filter
    return: the gaussian filter
    """
    gaussian_filter = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            gaussian_filter[i, j] = math.exp(
                -(i-size//2)**2/(2*sigma**2) - (j-size//2)**2/(2*sigma**2))
    return gaussian_filter


def applyGaussianFiltertoImage(image, sigma):
    """apply the gaussian filter to the image

    Args:
        image (mat): input image
        sigma (float): sigma of the gaussian filter

    Returns:
        mat: output image
    """
    gaussian_filter = createGaussianFilter(5, sigma)
    gaussian_image = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            gaussian_image[i, j] = np.sum(
                gaussian_filter*image[i-2:i+3, j-2:j+3])
    return gaussian_image


def convertImagefromRGBtoHSV(image):
    """convert the image from RGB to HSV"""
    image_HSV = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_HSV[i, j, 0] = image[i, j, 0]/255
            image_HSV[i, j, 1] = image[i, j, 1]/255
            image_HSV[i, j, 2] = image[i, j, 2]/255
    return image_HSV


def getVchannel(image_HSV):
    """generate the V channel of the image

    Args:
        image_HSV (mat): RGB image

    Returns:
        mat: V channel of the image
    """
    V = np.zeros(image_HSV.shape[0:2])
    for i in range(image_HSV.shape[0]):
        for j in range(image_HSV.shape[1]):
            V[i, j] = image_HSV[i, j, 2]
    return V


class checkUI:
    """check the uneven illumination of the image"""

    def __init__(self, image_path):
        """initialize the class

        Args:
            image_path (string): path of the image
        """
        self.image_path = image_path

    def get_result(self):
        """get the result of the check

        Returns:
            string: image path
        """
        return self.checkUnevenIllumination(self.image_path)

    def checkUnevenIllumination(self, image_path):
        """check the uneven illumination of the image

        Returns:
            string: image path
        """
        img = mpimg.imread(image_path)
        V = getVchannel(convertImagefromRGBtoHSV(img))
        img_gaussian = applyGaussianFiltertoImage(V, 1)
        result = V - img_gaussian
        plt.imshow(result)
        return result


def readImagefromFolder(folder_path):
    """read all the images from the folder

    Args:
        folder_path (string): path of the folder

    Returns:
        list: list of all the images
    """
    image_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_list.append(os.path.join(folder_path, filename))
    return image_list


def plotHistogram(array):
    """plot the histogram of the array
    array: the array to be plotted
    """
    plt.hist(array, bins=256)
    plt.show()


class classifyUI:
    """classify the image"""

    def __init__(self, image_path):
        """initialize the class

        Args:
            image_path (string): path of the image
        """
        self.image_path = image_path

    def get_result(self):
        """get the result of the classification

        Returns:
            string: image path
        """
        return self.classify(self.image_path)

    def calLuminanceMeantoRange(self, V):
        """calculate the luminance mean to range
        V: the V channel of the image
        return: the luminance mean to range
        """
        V_mean = np.mean(V)
        V_range = np.max(V) - np.min(V)
        return V_mean/V_range

    def classify(self, image_path):
        """classify the image

        Returns:
            string: image path
        """
        lmr_list = []
        for i in range(len(image_path)):
            img = mpimg.imread(image_path[i])
            V = getVchannel(convertImagefromRGBtoHSV(img))
            lmr = self.calLuminanceMeantoRange(V)
            lmr_list[i] = lmr
        return lmr_list


if __name__ == "__main__":
    checker = checkUI("./results/test.jpg")
    ui_mask = checker.get_result()
    img_list = readImagefromFolder("./results/test")
    classifer = classifyUI(img_list)
    ui_list = classifer.classify(img_list)
    plotHistogram(ui_list)
