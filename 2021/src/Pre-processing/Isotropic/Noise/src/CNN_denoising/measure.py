# import cv2
# import math
# import numpy
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def get_image_ssim(original_img, noisy_img):
    return ssim(original_img*255.0, noisy_img*255.0, data_range=original_img.max() - noisy_img.min(), multichannel=True)


# img_set: [batch, height, width, 3]
def get_set_psnr(original_img_set, noisy_img_set, img_height=64, img_width=64):
    psnr_sum = 0
    original_img_set = original_img_set.reshape(
        original_img_set.shape[0], img_height, img_width, 3)
    noisy_img_set = noisy_img_set.reshape(
        noisy_img_set.shape[0], img_height, img_width, 3)
    for i in range(original_img_set.shape[0]):
        psnr_sum += psnr(original_img_set[i], noisy_img_set[i], data_range=original_img_set[i].max(
        ) - noisy_img_set[i].min())
    return 1.0*psnr_sum/original_img_set.shape[0]


def get_set_ssim(originalSet, noisySet, img_height=64, img_width=64):
    ssim_sum = 0
    originalSet = originalSet.reshape(
        originalSet.shape[0], img_height, img_width, 3)
    noisySet = noisySet.reshape(noisySet.shape[0], img_height, img_width, 3)
    for i in range(originalSet.shape[0]):
        ssim_sum += ssim(originalSet[i], noisySet[i], data_range=originalSet[i].max(
        ) - noisySet[i].min(), multichannel=True)
    return 1.0*ssim_sum/originalSet.shape[0]
