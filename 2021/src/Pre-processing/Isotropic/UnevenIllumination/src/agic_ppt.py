from fileinput import filename
import numpy as np
import cv2
import glob
import os
import csv
from tqdm import tqdm
from functools import reduce


def neighbors(im, i, j, d=1):
    b = im[i-d:i+d+1, j-d:j+d+1].flatten()
    # remove the element (i,j)
    n = np.abs(np.hstack((b[:len(b)//2], b[len(b)//2+1:]))-im[i, j])
    return np.max(n)/im[i, j]


def estimateuneven(img):
    """
    Estimate the standard deviation of the image.

    Parameters
    ----------
    image : ndarray
        Image to estimate the standard deviation.

    Returns
    -------
    float
        The standard deviation of the image.

    """
    res = []
    illmask = cv2.medianBlur(img, 201)
    image = cv2.copyMakeBorder(illmask, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    width, height = image.shape
    for i in range(1, width-1):
        for j in range(1, height-1):
            if image[i, j]:
                res.append(neighbors(image, i, j))

    return Average(res),illmask


def Average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)


stdss = []
names = []

file_folder = '/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/src/img_ppt/agic/'

for file in tqdm(glob.glob("/home/nguyentansy/DATA/nguyentansy/PhD-work/Datasets/LVQ/uneven_illum/video1_*.avi")):
    cap = cv2.VideoCapture(file)
    names=os.path.basename(file)+str('.png')
    names2=str('img_ppt')+os.path.basename(file)+str('.png')
    names3=str('ori')+os.path.basename(file)+str('.png')
    print(names2)
    # Check if camera opened successfully
    if (cap.isOpened() is False):
        print("Error opening video stream or file")

    stds = []
    # Read until video is completed
    if(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret is True:
            # Display the resulting frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            std, illmask = estimateuneven(img[:, :, 2])
            # cv2.imshow('frame', illmask)
            # cv2.waitKey(0) 
            # cv2.destroyAllWindows() 
            # stds.append(std)
            # cv2.imwrite(file_folder+str(names), img[:, :, 2])
            # cv2.imwrite(file_folder+str(names2), illmask)
            cv2.imwrite(file_folder+str(names3), frame)
            # Press Q on keyboard to  exit
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break

        # Break the loop
    #     else:
    #         break
    # stdss.append(Average(stds))

# with open('agic.csv', 'w') as f:
#     writer = csv.writer(f, delimiter='\t')
#     writer.writerows(zip(names, stdss))