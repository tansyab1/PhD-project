import glob
import bm3d
import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity
from tqdm import tqdm
import math

noisy_dir = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Noise_var/test/input/'
PSNR = []
SSIM = []

dest_dir = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/BM3D/'
#  apply bm3d to all noisy images
# create a text file to store the results

text_file = open("bm3d.txt", "a")

# function to find the corresponding reference image


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
    # print(image.shape)
    width = image.shape[0]
    height = image.shape[1]
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
                   [0, 1, 0]], dtype=np.float64)
    L2 = np.array([[1, 0, 1],
                   [0, -4, 0],
                   [1, 0, 1]], dtype=np.float64)
    return L2 - 2*L1


def find_ref(noisy_path):
    # noisy_name = noisy_path.split('/')[-1]
    ref_name = noisy_path.replace('input', 'groundtruth')
    # ref_path = os.path.join(ref_dir, ref_name)
    return ref_name


for noisy_path in tqdm(glob.glob(os.path.join(noisy_dir, '*.jpg'))):
    noisy = cv2.imread(noisy_path)
    ref = cv2.imread(find_ref(noisy_path))
    basename = os.path.basename(noisy_path)
    desfile = os.path.join(dest_dir, basename)
    noise_sigma = estimateStandardDeviation(noisy)
    denoised = bm3d.bm3d(noisy, noise_sigma)
    cv2.imwrite(desfile, denoised)
    # calculate psnr and ssim
    PSNR.append(cv2.PSNR(ref, noisy))
    psnr = cv2.PSNR(ref, noisy)
    convert_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    convert_noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    SSIM.append(structural_similarity(convert_ref, convert_noisy))
    ssim = structural_similarity(convert_ref, convert_noisy)

    # write the results to the text file
    text_file.write(basename + ' PSNR: ' + str(psnr) +
                    ' SSIM: ' + str(ssim) + " level: " + str(noise_sigma)+"\n")

# move the text file to the destination folder
os.system('mv bm3d.txt ' + dest_dir)

print('PSNR: ', np.mean(PSNR))
print('SSIM: ', np.mean(SSIM))
