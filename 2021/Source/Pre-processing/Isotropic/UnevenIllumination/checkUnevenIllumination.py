import math
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


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


def checkUnevenIllumination(image_path):
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


if __name__ == "__main__":
    checkUnevenIllumination("./results/test.jpg")
