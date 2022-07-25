
import numpy as np
import cv2
import glob
import os
import seaborn as sns
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

def GlobalC_all(img, filter=None, name=None):
    # plt.figure()
    illmask = cv2.medianBlur(img, 201)
    # illmask = img.copy()
    gabor_img=gabor_filter(illmask, filter)
    
    image_global = illmask.copy()
    width, height = image_global.shape
    BsMat = np.zeros((width-6, height-6))
    
    for i in range(3, width-3):
        for j in range(3, height-3):
            if image_global[i, j]:
                g1,_ = localneighbors(image_global, i, j, d=2)
                g2,_ = localneighbors(image_global, i, j, d=3)
                Bs = (np.sum(g2)-np.sum(g1))/24
                BsMat[i-3, j-3] = gabor_img[i, j]/Bs
    
    # normalize the image
    # BsMat_normalize = BsMat/np.max(BsMat)

    # ax = sns.heatmap(BsMat_normalize)
    
    # # save the image to a file in the folder
    # BMat = np.where(BsMat > np.mean(BsMat), 1, 0)

    # print(np.count_nonzero(BMat))
    # get the folder link
    # folder_link = "/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/src/img_ppt/alc/test"
    # save the image to a file in the folder
    # plt.imsave(folder_link+"/GlobalC_"+name+".png", BsMat, cmap='gray')
    # plt.imsave(folder_link+"/GlobalC_bs"+name+".png", BsMat_normalize, cmap='gray', vmin=0, vmax=1)
    
    # save heatmap to a file in the folder
    # plt.savefig(folder_link+"/GlobalC_heatmap_"+name+".png")
    
    # save gambor image to a file in the folder
    # cv2.imwrite(folder_link+"/gabor_"+name+".png", gabor_img)
    
    # save the image to a file in the folder
    # create new plt.figure()
    # range 0.75 to 1.25 and from 0.9 to 1.1

    # select the sub array from BsMat where the value is greater than 0.9 and less than 1.1
    lim_down = BsMat.mean()- 4*BsMat.std()
    lim_up = BsMat.mean()+ 4*BsMat.std()
    BsMat_sub = np.array(BsMat[(BsMat < lim_up) & (BsMat > lim_down)])
    # print(BsMat_sub.shape)

    # plt.figure()
    # result = plt.hist(BsMat_sub.ravel(), range=(0.940, 1.060), color='c', edgecolor='k', bins=256)
    # plt.axvline(BsMat_sub.mean(), color='k', linestyle='dashed', linewidth=1)
    # min_ylim, max_ylim = plt.ylim()

    #set the ylim to 100
    # plt.ylim(0, 100)
    
    # plt.text(BsMat_sub.mean()*1.05, max_ylim*0.8, 'Mean: {:.4f}'.format(BsMat_sub.mean()))
    # plt.text(BsMat_sub.mean()*1.05, max_ylim*1.0, 'Std: {:.4f}'.format(BsMat_sub.std()))
    # plt.savefig(folder_link+"/fig_GlobalC_"+name+".png")

    # print(max(BsMat.ravel()))

    return np.std(
        
    )

def GlobalC(img, filter=None, name=None):
    plt.figure()
    illmask = cv2.medianBlur(img, 73)
    # illmask = img.copy()
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

    n_min, n_max = BsMat_normalize.min(), BsMat_normalize.max()

    ax = sns.heatmap(BsMat_normalize)
    
    # # save the image to a file in the folder
    BMat = np.where(BsMat > np.mean(BsMat), 1, 0)

    # print(np.count_nonzero(BMat))
    # get the folder link
    folder_link = "/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/src/img_ppt/alc/test"
    # save the image to a file in the folder
    plt.imsave(folder_link+"/GlobalC_"+name+".png", BMat, cmap='gray', vmin=0, vmax=1)
    # plt.imsave(folder_link+"/GlobalC_bs"+name+".png", BsMat_normalize, cmap='gray', vmin=0, vmax=1)
    
    # save heatmap to a file in the folder
    plt.savefig(folder_link+"/GlobalC_heatmap_"+name+".png")
    
    # save gambor image to a file in the folder
    cv2.imwrite(folder_link+"/gabor_"+name+".png", gabor_img)
    
    # save the image to a file in the folder
    # create new plt.figure()
    plt.figure()
    result = plt.hist(BsMat_normalize.ravel(), range=(n_min, n_max), color='c', edgecolor='k', alpha=0.65, bins=128)
    plt.axvline(BsMat_normalize.mean(), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(BsMat_normalize.mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(BsMat_normalize.mean()))
    plt.savefig(folder_link+"/fig_GlobalC_"+name+".png")

    return np.mean(BsMat)


def LocalC(img, d=1, name=None):
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

    illmask = cv2.medianBlur(img, 72)
    # illmask = img.copy()
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
    BMat = BsMat/np.max(BsMat)

    # print(BMat)
    # binarize the image with a threshold of 0.5
    BcMat = np.where(BsMat > np.mean(BsMat), 1, 0)
    # save the image to a file in the folder
    # get the folder link
    folder_link = "/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/src/img_ppt/alc"
    # save the image to a file in the folder
    cv2.imwrite(folder_link+"/LocalC_bs"+name+".png", BMat*255)
    # cv2.imwrite(folder_link+"/LocalC_"+name+".png", BcMat*255)

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

    for file in tqdm(glob.glob("/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/src/img_ppt/alc/input/case 1/*.png")):
        # print(file)
        name=os.path.basename(file)
        img = cv2.imread(file)
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        GlobalC_level = GlobalC_all(img[:, :, 2], filter=filter, name = name)
        # LocalC_level = LocalC(img[:, :, 2], name = name)

    # with open('test_artificial.csv', 'w') as f:
    #     writer = csv.writer(f, delimiter='\t')
    #     writer.writerows(zip(names, stdss))
