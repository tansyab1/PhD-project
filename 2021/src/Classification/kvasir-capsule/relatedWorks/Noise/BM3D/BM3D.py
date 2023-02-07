import glob
import bm3d
import numpy as np
import cv2
import os

noisy_dir = 'noisy'
PSNR = []
SSIM = []

#  apply bm3d to all noisy images

for noisy_path in glob.glob(os.path.join(noisy_dir, '*.png')):
    noisy = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
    denoised = bm3d.bm3d.bm3d(noisy, 1.0)
    cv2.imwrite(noisy_path.replace('noisy', 'bm3d'), denoised)
    # calculate psnr and ssim
    PSNR.append(cv2.PSNR(denoised, noisy))
    SSIM.append(cv2.SSIM(denoised, noisy))

print('PSNR: ', np.mean(PSNR))
print('SSIM: ', np.mean(SSIM))
