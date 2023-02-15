import cv2
# import os
import seaborn as sns
from tqdm import tqdm
import numpy as np
import glob
import os
from skimage.transform import warp_polar
import matplotlib.pyplot as plt


def detectBlur(img_path, threhold):
    """detect blur of the image

    Args:
        img_path (string): image path
        threhold (float): threshold of blur

    Returns:
        Bool: yes or no
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = getVarianofLaplace(gray)
    if (var < threhold):
        return True
    else:
        return False


def readImagefromFolder(folder="/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/UI_var/test/input"):
    sigma_list2 = []
    file = open("sigma_list_UI.txt", "w")
    # for filename in tqdm(glob.glob("%s/*/*/*/*" % folder)):
    #     img = cv2.imread(filename, 0)
    #     sigma_n = getVarianofLaplace(img)
    #     sigma_list1.append(sigma_n)

    for filename in tqdm(glob.glob("%s/*.jpg" % folder)):
        img = cv2.imread(filename)
        # sigma_n = getVarianofLaplace(img)
        # sigma_list2.append(sigma_n)
        sigma_n = getIHED(img)
        file.write(os.path.basename(filename) + " " + str(sigma_n) + "\n")
    
    return sigma_list2


def plotHistogram(arr2):
    """
    Plot the histogram of the array.

    Parameters
    ----------
    arr : ndarray
        Array to plot the histogram.

    """
    # plt.hist(arr.ravel(), 256, [0, 256])
    # plt.show()

    plt.hist(arr2, bins=8)
    plt.show()

    plt.title('Variance of Laplace analysis', fontsize=22)
    plt.legend()
    # filesave = "./varLaphist.png"
    # plt.savefig(filesave)


def getVarianofLaplace(img):
    """get variance of laplace of the image
    Args:
        img (numpy array): image
    Returns:
        float: variance of laplace
    """
    return cv2.Laplacian(img, cv2.CV_64F).var()

# Discret fourier transform of image
def getDFT(img):
    """get DFT of the image
    Args:
        img (numpy array): image
    Returns:
        numpy array: DFT of the image
    """
    return cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

# convert image RGB to gray
def convertRGBtoGray(img):
    """convert image RGB to gray
    Args:
        img (numpy array): image
    Returns:
        numpy array: gray image
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# convert image gray from cartesian to polar coordinate system
def convertCartesianToPolar(img):
    """convert image gray from cartesian to polar coordinate system
    Args:
        img (numpy array): image gray
    Returns:
        numpy array: polar image gray
    """
    return warp_polar(img, scaling='linear')

# calculate the total radial energy of the image
def getRadialEnergy(img):
    """calculate the total radial energy of the image
    Args:
        img (numpy array): image gray
    Returns:
        float: total radial energy of the image
    """
    return np.sum(img, axis=0) / img.shape[0]

# calcuate the global blur of the image
def getGlobalBlur(img):
    """calculate the global blur of the image
    Args:
        img (numpy array): image gray
    Returns:
        float: global blur of the image
    """
    gray_img = convertRGBtoGray(img)
    gray_img_polar = convertCartesianToPolar(gray_img)
    img_polar = convertCartesianToPolar(img)
    radial_energy_gray = getRadialEnergy(gray_img_polar)
    radial_energy_img = getRadialEnergy(img_polar)
    return np.log(np.sum(np.abs(radial_energy_img - radial_energy_gray)) / np.max(radial_energy_img.shape))

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
    




if __name__ == '__main__':
    sigma_list2 = readImagefromFolder()
    print("sigma_list1: ", sigma_list2)
    plotHistogram(sigma_list2)
