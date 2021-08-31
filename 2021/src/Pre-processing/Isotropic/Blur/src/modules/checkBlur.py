import cv2
# import os
import seaborn as sns
from tqdm import tqdm
import numpy as np
import glob
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


def readImagefromFolder(folder="/home/nguyentansy/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/data/labeled-images/"):
    sigma_list1 = []
    sigma_list2 = []
    for filename in tqdm(glob.glob("%s/*/*/*/*" % folder)):
        img = cv2.imread(filename, 0)
        sigma_n = getVarianofLaplace(img)
        sigma_list1.append(sigma_n)

    for filename in tqdm(glob.glob("%s/*/pathological-findings/*/*" % folder)):
        img = cv2.imread(filename, 0)
        sigma_n = getVarianofLaplace(img)
        sigma_list2.append(sigma_n)
    return sigma_list1, sigma_list2


def plotHistogram(arr, arr2):
    """
    Plot the histogram of the array.

    Parameters
    ----------
    arr : ndarray
        Array to plot the histogram.

    """
    # plt.hist(arr.ravel(), 256, [0, 256])
    # plt.show()

    # Draw Plot
    plt.figure(figsize=(13, 10), dpi=80)
    sns.histplot(arr, color="g", label="labeled images")
    sns.histplot(arr2, label="pathological findings", color="orange")
    # plt.ylim(0, 0.35)
    # plt.xticks(np.arange(0, 2, 0.05), rotation=45)
    # Decoration
    plt.title('Variance of Laplace analysis', fontsize=22)
    plt.legend()
    filesave = "/home/nguyentansy/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/Blur/src/results/varLaphist.png"
    plt.savefig(filesave)


def getVarianofLaplace(img):
    """get variance of laplace of the image
    Args:
        img (numpy array): image
    Returns:
        float: variance of laplace
    """
    return cv2.Laplacian(img, cv2.CV_64F).var()


if __name__ == '__main__':
    sigma_list1, sigma_list2 = readImagefromFolder()
    plotHistogram(sigma_list1, sigma_list2)
