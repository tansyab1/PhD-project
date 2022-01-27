
import cv2
import numpy as np
from skimage.util import random_noise
import os


def create_noise(image, var, mean=0):
    noise_img = random_noise(image, mode='gaussian',
                             mean=mean, var=var/255, clip=True)
    return np.array(255*noise_img, dtype='uint8')


if __name__ == "__main__":
    save_path = '/home/nguyentansy/PhD-work/PhD-project/2021/Source/Pre-processing/Isotropic/results/AWGN/'
    os.makedirs(save_path, exist_ok=True)
    sigmas = [0.0005, 0.001, 0.005, 0.01]
    img = cv2.imread('results/test.jpg')
    for sigma in sigmas:
        # cv2.imshow('Original',img)
        file_name = 'AWGN'+'_var:'+str(sigma)+'.png'
        completeName = os.path.join(save_path, file_name)
        output = create_noise(img, var=sigma)
        # cv2.imshow('Motion Blur', output)
        cv2.imwrite(completeName, output)
        # cv2.waitKey(0)
