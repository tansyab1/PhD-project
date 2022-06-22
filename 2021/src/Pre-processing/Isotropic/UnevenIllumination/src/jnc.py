
import numpy as np
import cv2
import glob
import os
import csv
from tqdm import tqdm
from matplotlib import pyplot as plt
from functools import reduce


# function to read mask
def read_mask(mask_path="src/Pre-processing/Isotropic/UnevenIllumination/src/mask.png"):
    mask = cv2.imread(mask_path, 0)
    mask = mask.astype(np.bool)
    return mask

def apply_mask(img, mask):
    """
    Apply a mask to an image.

    Parameters
    ----------
    img : ndarray
        Image to apply the mask to.
    mask : ndarray
        Mask to apply to the image.

    Returns
    -------
    ndarray
        Image with the mask applied.

    """

    return img * mask


def localneighbors(im, i, j, d=1):
    # print(d, i, j)
    b = im[i-d:i+d+1, j-d:j+d+1].flatten()
    gmm = (np.sum(b, dtype=np.float64)-im[i, j])/(8*d)
    return b, gmm

def create_gaborfilter():
    # This function is designed to produce a set of GaborFilters 
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree
     
    filters = []
    num_filters = 4
    ksize = 7  # The local area to evaluate
    sigma = 1 # Larger Values produce more edges
    lambd = 1
    gamma = 1.88
    psi = 0  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters

def gabor_filter(img, filters):
    """
    Apply Gabor filters to the image.

    Parameters
    ----------
    img : ndarray
        Image to apply the Gabor filters.
    filters : list
        List of Gabor filters.

    Returns
    -------
    ndarray
        Gabor filtered image.
    """
    
    gabor_img = np.zeros(img.shape)
    for kern in filters:
        gabor_img += np.abs(cv2.filter2D(img, cv2.CV_64F, kern, borderType=cv2.BORDER_REFLECT))
    return gabor_img/4

def jnc_plus(img, filter=None, name=None):
    plt.figure()
    illmask = cv2.medianBlur(img, 201)
    gabor_img=gabor_filter(illmask, filter)
    BsMat = np.zeros(img.shape)
    image_global = cv2.copyMakeBorder(illmask, 3, 3, 3, 3, cv2.BORDER_REFLECT)
    width, height = image_global.shape
    
    for i in range(3, width-3):
        for j in range(3, height-3):
            if image_global[i, j]:
                g1,_ = localneighbors(image_global, i, j, d=2)
                g2,_ = localneighbors(image_global, i, j, d=3)
                Bs = (np.sum(g2)-np.sum(g1))/24
                BsMat[i-3, j-3] = gabor_img[i-3, j-3]/Bs
    
    # normalize the image
    BsMat_normalize = BsMat/np.max(BsMat)
    
    # save the image to a file in the folder
    BMat = np.where(BsMat > np.mean(BsMat), 1, 0)
    # get the folder link
    folder_link = "/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/src/img_ppt/alc"
    # save the image to a file in the folder
    plt.imsave(folder_link+"/jnc_plus_"+name+".png", BMat, cmap='gray', vmin=0, vmax=1)
    # save gambor image to a file in the folder
    cv2.imwrite(folder_link+"/gabor_"+name+".png", gabor_img)
    
    result = plt.hist(BsMat_normalize.ravel(), color='c', edgecolor='k', alpha=0.65)
    plt.axvline(BsMat_normalize.mean(), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(BsMat_normalize.mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(BsMat_normalize.mean()))
    plt.savefig(folder_link+"/fig_jnc_plus_"+name+".png")

    return np.mean(BsMat)


def jnc(img, d=1, name=None):
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

    illmask = cv2.medianBlur(img, 201)
    BsMat = np.zeros(img.shape)
    res = np.zeros(img.shape)
    # BgMat = np.zeros(img.shape)
    image = cv2.copyMakeBorder(illmask, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    # image_global = cv2.copyMakeBorder(illmask, d, d, d, d, cv2.BORDER_REFLECT)
    width, height = image.shape
    Bg = np.mean(illmask)
    # width_global, height_global = image_global.shape

    # for i in range(d, width_global-d):
    #     for j in range(d, height_global-d):
    #         if image_global[i, j]:
    #             Bg, _ = localneighbors(image_global, i, j, d=3)
    #             BgMat[i-d, j-d] = Bg
    for i in range(1, width-1):
        for j in range(1, height-1):
            if image[i, j]:
                g, Bs = localneighbors(image, i, j, d=1)
                Ba = 0.923*Bs + 0.077*Bg
                B_top = np.abs(g.astype(float)-float(Ba))
                BsMat[i-1, j-1] = np.max(B_top/Ba)

    # normalize the image
    # BMat = BsMat/np.max(BsMat)
    # binarize the image with a threshold of 0.5
    BMat = np.where(BsMat > np.mean(BsMat), 1, 0)
    # save the image to a file in the folder
    # get the folder link
    folder_link = "/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/src/img_ppt/alc"
    # save the image to a file in the folder
    cv2.imwrite(folder_link+"/jnc_"+name+".png", BMat*255)

    return np.mean(BsMat)


def Average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)


if __name__ == '__main__':
    #     img = np.ones((7, 7))
    #     a, b = localneighbors(img, 0, 0, d=1)
    #     print(b)

    stdss = []
    names = []
    filter = create_gaborfilter()

    for file in tqdm(glob.glob("/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/src/img_ppt/alc/*.png")):
        # print(file)
        name=os.path.basename(file)
        img = cv2.imread(file)
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        jnc_plus_level = jnc_plus(img[:, :, 2], filter=filter, name = name)
        # jnc_level = jnc(img[:, :, 2], name = name)
        print(jnc_plus_level)

    # with open('test_artificial.csv', 'w') as f:
    #     writer = csv.writer(f, delimiter='\t')
    #     writer.writerows(zip(names, stdss))
