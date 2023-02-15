import os
import cv2
from glob import glob

def downscale(img, scale_percent):
    # calculate the 50 percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    # dsize
    dsize = (width, height)
    # resize image
    output = cv2.resize(img, dsize)
    return output

def upscale(img, scale_percent):
    # calculate the 50 percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    # dsize
    dsize = (width, height)
    # resize image
    output = cv2.resize(img, dsize)
    return output

if __name__ == "__main__":
    img_path = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/resultsforPaper/ui/mir"
    for filename in glob("%s/*" % img_path):
        print(filename)
        img=cv2.imread(filename)
        # convert to hsv    
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 2] = 0.9* img_hsv[:, :, 2]
        # convert to rgb
        out = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(filename+".png", out)
        