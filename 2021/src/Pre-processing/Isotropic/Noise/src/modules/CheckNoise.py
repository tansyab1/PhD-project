import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
# import os
# import pandas as pd
import glob
import seaborn as sns
from tqdm import tqdm
import matplotlib as mpl
# import warnings; warnings.filterwarnings(action='once')

large = 22
med = 16
small = 12
params = {'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
# %matplotlib inline

# Version
print(mpl.__version__)  # > 3.0.0
print(sns.__version__)  # > 0.9.0


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


def readImagefromFolder(folder="/home/nguyentansy/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/data/labeled-images/"):
    sigma_list = []
    for filename in tqdm(glob.glob("%s/*/pathological-findings/*/*" % folder)):
        img = cv2.imread(filename, 0)
        sigma_n = estimateStandardDeviation(img)
        sigma_list.append(sigma_n)
    return sigma_list


def plotHistogram(arr):
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
    sns.histplot(arr, color="g",
                 label="noise standard deviation")
    # plt.ylim(0, 0.35)
    plt.xticks(np.arange(0, 1.5, 0.05), rotation=45)
    # Decoration
    plt.title('Noise standard deviation analysis', fontsize=22)
    plt.legend()
    filesave = "/home/nguyentansy/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/Noise/src/denoising_rgb/results/sigmahist.png"
    plt.savefig(filesave)


if __name__ == "__main__":
    img_list = readImagefromFolder()
    plotHistogram(img_list)
