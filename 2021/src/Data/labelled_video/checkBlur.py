import cv2
# import os
import seaborn as sns
from tqdm import tqdm
import numpy as np
import glob
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

# Discret fourier transform of image
def getDFT(img):
    """get DFT of the image
    Args:
        img (numpy array): image
    Returns:
        numpy array: DFT of the image
    """
    dft=cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    return magnitude_spectrum

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

# binominal filteing of the image
def binomialFilter(img):
    """binominal filteing of the image
    Args:
        img (numpy array): image gray
        kernal_size (int): size of the kernal
    Returns:
        numpy array: binominal filteing of the image
    """
    # Create a binomial kernel
    kernel = np.array([[0.0625, 0.125, 0.0625],
                        [0.125, 0.25, 0.125],
                        [0.0625, 0.125, 0.0625]])
    # Apply the filter
    return cv2.filter2D(img, -1, kernel)


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
    gray_img_norminal = binomialFilter(gray_img)

    # fourier transform of the image
    dft_img = getDFT(gray_img)
    # print(dft_img.shape)
    dft_img_norminal = getDFT(gray_img_norminal)

    # convert image gray from cartesian to polar coordinate system
    img_polar = convertCartesianToPolar(dft_img)
    img_polar_norminal = convertCartesianToPolar(dft_img_norminal)

    # calculate the total radial energy of the image
    radial_energy_gray = getRadialEnergy(img_polar)
    radial_energy_gray_norminal = getRadialEnergy(img_polar_norminal)

    return np.log(np.sum(np.abs(radial_energy_gray - radial_energy_gray_norminal)) / np.max(radial_energy_gray.shape))


    




if __name__ == '__main__':
    sigma_list1, sigma_list2 = readImagefromFolder()
    plotHistogram(sigma_list1, sigma_list2)
