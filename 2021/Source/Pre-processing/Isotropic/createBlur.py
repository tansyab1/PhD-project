#!/home/nguyentansy/.virtualenvs/PhD-AI/bin/python3

import cv2
import numpy as np
import os
#size - in pixels, size of motion blur
#angel - in degrees, direction of motion blur
def apply_motion_blur(image, size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )  
    k = k * ( 1.0 / np.sum(k) )  
    # cv2.imshow('kernel', k)   
    # cv2.waitKey(0)   
    return cv2.filter2D(image, -1, k) 

def apply_defocus_blur(image, sigma):
    dst = cv2.GaussianBlur(image,(int(6*sigma)+1,int(6*sigma)+1),sigma)
    return dst


if __name__ == '__main__': 
    # save_path = '/home/nguyentansy/PhD-work/PhD-project/2021/Source/Pre-processing/Isotropic/results/blur/Motion/'
    # os.makedirs(save_path, exist_ok=True)
    # angles=[0,45,90,135]
    # levels=[5,10,15,20]
    # img = cv2.imread('results/test.jpg')
    # for angle in angles:
    #     for level in levels:
    #         # cv2.imshow('Original',img)
    #         file_name ='motion'+'_level:'+str(level)+'_angle:'+str(angle)+'.png'
    #         completeName = os.path.join(save_path, file_name)
    #         output = apply_motion_blur(img,level,angle)
    #         # cv2.imshow('Motion Blur', output)
    #         cv2.imwrite(completeName,output)
    #         # cv2.waitKey(0)

    save_path = '/home/nguyentansy/PhD-work/PhD-project/2021/Source/Pre-processing/Isotropic/results/Blur/Defocus/'
    os.makedirs(save_path, exist_ok=True)
    sigmas=[0.75,1,2,3]
    img = cv2.imread('results/test.jpg')
    for sigma in sigmas:
        # cv2.imshow('Original',img)
        file_name ='defocus'+'_sigma:'+str(sigma)+'.png'
        completeName = os.path.join(save_path, file_name)
        output = apply_defocus_blur(img,sigma)
        # cv2.imshow('Motion Blur', output)
        cv2.imwrite(completeName,output)
        # cv2.waitKey(0)
