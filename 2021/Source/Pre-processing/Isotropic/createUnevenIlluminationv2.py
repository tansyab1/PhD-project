import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math
import os
import sys
np.set_printoptions(threshold=sys.maxsize)

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
        before= hsv[:, :, 2]+255

        # hsv[:, :, 2] = np.where(hsv[:, :, 2]>30,hsv[:, :, 2] * self.transparency + mask * (1 - self.transparency),hsv[:, :, 2])
        hsv[:, :, 2] = np.where(hsv[:, :, 2]>30,mask/255*hsv[:, :, 2],hsv[:, :, 2])

        after= hsv[:, :, 2]

        hsv_res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        hsv_res[hsv_res > 255] = 255

        hsv_res = np.asarray(hsv_res, dtype=np.uint8)

        return hsv_res,mask,before,after


if __name__ == "__main__":
    columns = 4
    rows = 4
    i=1
    fig=plt.figure(figsize=(20, 20))
    circle_light_shape=[500,300]
    max_intensities = [50,100,180,250]
    modes = ['circle','oval']
    thetas = [0,45,90,135]
    transparency = np.random.uniform(0.2, 0.3)
    frame = cv2.imread('/content/drive/MyDrive/Colab Notebooks/PhD-project/Pre-processing/Isotropic/test.jpg')
    centers=[[np.shape(frame)[0]//2,np.shape(frame)[1]//2]]

    for max_intensity in max_intensities:
        for center in centers:
            frame = cv2.imread('/content/drive/MyDrive/Colab Notebooks/PhD-project/Pre-processing/Isotropic/test.jpg')
            file_name ='UnevenIllumination'+'_mode:circle'+'_level:'+str(max_intensity)+'_center:'+str(center)
            p = createUnevenIllumination(circle_light_shape,max_intensity,transparency,mode='circle')
            hsv_res,mask,before,after = p.createUnevenIllumination(frame,center)

            fig.add_subplot(rows, columns, i)
            plt.imshow(hsv_res)

            fig.add_subplot(rows, columns, i+columns)
            plt.imshow(mask,cmap='gray',vmin=0, vmax=255)
            
            fig.add_subplot(rows, columns, i+3*columns)
            plt.imshow(after,cmap='gray',vmin=0, vmax=255)
            
            fig.add_subplot(rows, columns, i+2*columns)
            plt.imshow(before-255,cmap='gray',vmin=0, vmax=255)
            
            i+=1
    fig.tight_layout()
    plt.show()
