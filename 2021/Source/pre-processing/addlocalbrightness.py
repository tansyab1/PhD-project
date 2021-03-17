import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
arr = np.zeros((256,256), dtype=np.uint8)
imgsize = arr.shape[:2]
innerColor = (0)
outerColor = (255)
for y in range(imgsize[1]):
    for x in range(imgsize[0]):
        #Find the distance to the center
        distanceToCenter = np.sqrt((x - imgsize[0]//2) ** 2 + (y - imgsize[1]//2) ** 2)

        #Make it on a scale from 0 to 1innerColor
        distanceToCenter = distanceToCenter / (np.sqrt(2) * imgsize[0]/2)

        #Calculate r, g, and b values
        r = outerColor * distanceToCenter + innerColor * (1 - distanceToCenter)
        # g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)
        # b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)
        # print r, g, b
        arr[y, x] = int(r)


img = cv2.imread('test.jpg') #load rgb image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv

print(hsv[:256,:256,2].shape)
hsv[:256,:256,2]=hsv[0:256,0:256,2]*arr
img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite("image_processed.jpg", img)