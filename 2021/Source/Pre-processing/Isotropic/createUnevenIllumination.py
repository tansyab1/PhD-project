import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math


class createUnevenIllumination:
    def __init__(self,gradient_shape,max_intensity,transparency,mode):
        self.gradient_shape = gradient_shape
        self.max_intensity = max_intensity
        self.transparency = transparency
        self.mode = mode



    def create_oval(self,image, position, theta):
        st, ct =  math.sin(theta), math.cos(theta)
        aa, bb = self.gradient_shape[0]**2, self.gradient_shape[1]**2

        weights = np.zeros((np.shape(image)[0],np.shape(image)[1]), np.float64)    
        for x in range(np.shape(image)[0]):
            for y in range(np.shape(image)[1]):
                weights[x,y] = ((((x-position[0]) * ct + (y-position[1]) * st) ** 2) / aa
                    + (((x-position[0]) * st - (y-position[1]) * ct) ** 2) / bb)
                
        return np.clip(1.0 - weights, 0, 1)* self.max_intensity



    def create_mask(self,given_size,center,gradient_shape):
        mask = np.zeros((gradient_shape,gradient_shape), dtype=np.uint8)
        imgsize = mask.shape[:2]

        # scale of intensity
        innerColor = 0
        outerColor = self.max_intensity

        for y in range(imgsize[1]):
            for x in range(imgsize[0]):
                #Find the distance to the center
                distanceToCenter = np.sqrt((x - imgsize[0]//2) ** 2 + (y - imgsize[1]//2) ** 2)

                #Make it on a scale from 0 to 1innerColor
                distanceToCenter = distanceToCenter / (np.sqrt(1) * imgsize[0]/2)

                #Calculate intensity values
                intensity = innerColor * distanceToCenter + outerColor * (1 - distanceToCenter)
                if distanceToCenter < 1:
                    mask[y, x] = int(intensity)

        result = np.zeros((given_size[0],given_size[1]),dtype=np.uint8)

        ori_left= [x-int(gradient_shape//2) for x in center]
        ori_right= [x+int(gradient_shape//2) for x in center]

        #result image
        left_corner = [center[i]-int(gradient_shape//2) if center[i]-int(gradient_shape//2) >= 0 else 0 for i in range(2)]
        right_corner = [center[i]+int(gradient_shape//2) if center[i]+int(gradient_shape//2) <= given_size[i] else given_size[i] for i in range(2)]

        # mask
        mask_left = [left_corner[i] - ori_left[i] for i in range(len(left_corner))] 
        mask_right = [right_corner[i] - ori_right[i] + gradient_shape for i in range(len(right_corner))] 
        result[left_corner[0]:right_corner[0],left_corner[1]:right_corner[1]]= mask[mask_left[0]:mask_right[0],mask_left[1]:mask_right[1]]

        return result



    def createUnevenIllumination(self,image,center,theta=40):
        if self.mode =='circle':
            mask = self.create_mask([np.shape(image)[0],np.shape(image)[1]],center, self.gradient_shape[0])+255-self.max_intensity
        elif self.mode =='oval':
            mask = self.create_oval(image,center,theta)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        image[:, :, 0] = np.where(image[:, :, 0]>30,image[:, :, 0] * self.transparency + mask * (1 - self.transparency),image[:, :, 0])
        image[:, :, 1] = np.where(image[:, :, 1]>30,image[:, :, 1] * self.transparency + mask * (1 - self.transparency),image[:, :, 1])
        image[:, :, 2] = np.where(image[:, :, 2]>30,image[:, :, 2] * self.transparency + mask * (1 - self.transparency),image[:, :, 2])

        hsv[:, :, 2] = np.where(hsv[:, :, 2]>30,hsv[:, :, 2] * self.transparency + mask * (1 - self.transparency),hsv[:, :, 2])
        lab[:, :, 0] = np.where(lab[:, :, 0]>30,lab[:, :, 0] * self.transparency + mask * (1 - self.transparency),lab[:, :, 0])
        
        hsv_res = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        lab_res = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)

        hsv_res[hsv_res > 255] = 255
        lab_res[lab_res > 255] = 255

        hsv_res = np.asarray(hsv_res, dtype=np.uint8)
        lab_res = np.asarray(lab_res, dtype=np.uint8)

        return image,hsv_res,lab_res


if __name__ == "__main__":

    given_size = [300,500]
    center=[200,200]
    circle_light_shape=[200,300]
    max_intensity = 200
    mode = 'oval'
    theta = 40
    transparency = np.random.uniform(0.2, 0.3)
    
    frame = cv2.imread('results/test.jpg')
    p = createUnevenIllumination(circle_light_shape,max_intensity,transparency,mode)
    image,hsv_res,lab_res = p.createUnevenIllumination(frame,center,theta)
    
    cv2.imshow('HSV',hsv_res)
    cv2.imshow('Lab',lab_res)
    cv2.imshow('RGB',image)
    # cv2.imwrite('createUnevenIllumination2.png',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
