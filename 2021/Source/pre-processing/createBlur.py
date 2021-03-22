
import cv2
import numpy as np

#size - in pixels, size of motion blur
#angel - in degrees, direction of motion blur
def apply_motion_blur(image, size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )  
    k = k * ( 1.0 / np.sum(k) )        
    return cv2.filter2D(image, -1, k) 

def apply_defocus_blur(image, size):
    return cv2.GaussianBlur(img,size,0)


if __name__ == '__main__': 
    
    img = cv2.imread('test.jpg')
    cv2.imshow('Original',img)
    output = apply_motion_blur(img,12,30)
    cv2.imshow('Motion Blur', output)
    cv2.waitKey(0)
