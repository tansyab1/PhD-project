import numpy as np
# import tarfile
import cv2
import math
import os
# import matplotlib.pyplot as plt
from skimage.util import random_noise
import glob
# import seaborn as sns
from tqdm import tqdm
from createAWGN import create_noise
# import matplotlib as mpl
# import pywt

# large = 22
# med = 16
# small = 12
# params = {'legend.fontsize': med,
#           'figure.figsize': (16, 10),
#           'axes.labelsize': med,
#           'axes.titlesize': med,
#           'xtick.labelsize': med,
#           'ytick.labelsize': med,
#           'figure.titlesize': large}
# plt.rcParams.update(params)
# plt.style.use('seaborn-whitegrid')
# sns.set_style("white")
# # %matplotlib inline

# # Version
# print(mpl.__version__)  # > 3.0.0
# print(sns.__version__)  # > 0.9.0


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
                   [0, 1, 0]], dtype=float)
    L2 = np.array([[1, 0, 1],
                   [0, -4, 0],
                   [1, 0, 1]], dtype=float)
    return L2 - 2*L1


def check_noise(file_path, thresh=0.6):
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
    img_noise = []
    stds = []
    img = cv2.imread(file_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sigma_n = estimateStandardDeviation(img_gray)
    if sigma_n < thresh:
        for i in range(5, 25, 5):
            img_noise.append(create_noise(img, var=i/255))
            stds.append(i)

    return img, img_noise, stds


save_path = '/home/nguyentansy/DATA/PhD-work/Datasets/create/Noise'


def createNoise(folder="/home/nguyentansy/DATA/PhD-work/Datasets/create/"):
    img_noise = []
    for filename in tqdm(glob.glob("%s/*/*/*/*" % folder)):
        img, img_noise, stds = check_noise(filename)
        base_name = os.path.basename(filename)
        if img_noise:
            cv2.imwrite("%s/%s/%s" %
                            (save_path,'reference', base_name), img)
            for i in range(4):
                cv2.imwrite("%s/%s/%s" %
                            (save_path, str(stds[i]), base_name), img_noise[i])
    # return


def apply_motion_blur(image, size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[(size-1) // 2, :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D(
        (size / 2 - 0.5, size / 2 - 0.5), angle, 1.0), (size, size))
    k = k * (1.0 / np.sum(k))
    # cv2.imshow('kernel', k)
    # cv2.waitKey(0)
    return cv2.filter2D(image, -1, k)


def apply_defocus_blur(image, sigma):
    dst = cv2.GaussianBlur(image, (int(6*sigma)+1, int(6*sigma)+1), sigma)
    return dst


def detectBlur(img_path, threhold=800):
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
    if (var > threhold):
        return True
    else:
        return False


def getVarianofLaplace(img):
    """get variance of laplace of the image
    Args:
        img (numpy array): image
    Returns:
        float: variance of laplace
    """
    return cv2.Laplacian(img, cv2.CV_64F).var()


if __name__ == '__main__':
    createNoise()
