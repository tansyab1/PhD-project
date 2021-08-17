import cv2
import math
import numpy
from skimage.measure._structural_similarity import compare_ssim as ssim


def get_image_ssim(original_img,noisy_img):
    return ssim(original_img*255.0, noisy_img*255.0,data_range=original_img.max() - noisy_img.min(), multichannel=False)

def get_set_ssim(originalSet,noisySet,img_height=64, img_width=64):
    ssim_sum = 0
    originalSet = originalSet.reshape(originalSet.shape[0],img_height, img_width, 1)
    noisySet = noisySet.reshape(noisySet.shape[0],img_height, img_width, 1)
    for i in range(originalSet.shape[0]):
        ssim_sum += ssim(originalSet[i], noisySet[i],data_range=originalSet[i].max() - noisySet[i].min(), multichannel=True)
    return 1.0*ssim_sum/originalSet.shape[0]
