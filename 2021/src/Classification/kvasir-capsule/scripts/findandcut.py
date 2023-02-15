import cv2 as cv
# import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import os

def templatematch(img, template):
    w, h = img.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv.TM_SQDIFF_NORMED']
    for meth in methods:
        method = eval(meth)
        # Apply template Matching
        res = cv.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        return top_left, bottom_right


pathimg = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/resultsforPaper/noise/mirnet"
pathgt = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/resultsforPaper/noise/uformer"
pathbm3d = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/resultsforPaper/noise/bm3d"
pathcycleisp = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/resultsforPaper/noise/cycleisp"
pathdanet = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/resultsforPaper/noise/danet"
pathmprnet = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/resultsforPaper/noise/mprnet"
pathproposed = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/resultsforPaper/noise/proposed"
pathridnet = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/resultsforPaper/noise/ridnet"
pathvdnet = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/resultsforPaper/noise/vdnet"

for filename in glob("%s/*.png" % pathimg):
    template = cv.imread(filename.replace("mirnet", "uformer"), cv.IMREAD_GRAYSCALE)
    top_left, bottom_right = templatematch(cv.imread(filename, cv.IMREAD_GRAYSCALE), template)

    for file in glob("%s/*" % pathgt):
        if filename.split("/")[-1] == file.split("/")[-1]:
            img = cv.imread(file)
            crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            cv.imwrite(os.path.join(pathgt, file.split("/")[-1]+"_crop.png"), crop_img)
            
    for file in glob("%s/*" % pathbm3d):
        # print("filename: ", filename)
        # print("file: ", file)
        if os.path.basename(filename).split(".")[0] == os.path.basename(file).split(".")[0]:
            
            img = cv.imread(file)
            crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            cv.imwrite(os.path.join(pathbm3d, file.split("/")[-1]+"_crop.png"), crop_img)
            
    for file in glob("%s/*" % pathcycleisp):
        if os.path.basename(filename).split(".")[0] == os.path.basename(file).split(".")[0]:
            img = cv.imread(file)
            crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            cv.imwrite(os.path.join(pathcycleisp, file.split("/")[-1]+"_crop.png"), crop_img)
            
    for file in glob("%s/*" % pathdanet):
        if os.path.basename(filename).split(".")[0] == os.path.basename(file).split(".")[0]:
            img = cv.imread(file)
            crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            cv.imwrite(os.path.join(pathdanet, file.split("/")[-1]+"_crop.png"), crop_img)
    
    for file in glob("%s/*" % pathmprnet):
        if os.path.basename(filename).split(".")[0] == os.path.basename(file).split(".")[0]:
            img = cv.imread(file)
            crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            cv.imwrite(os.path.join(pathmprnet, file.split("/")[-1]+"_crop.png"), crop_img)
            
    for file in glob("%s/*" % pathproposed):
        if os.path.basename(filename).split(".")[0] == os.path.basename(file).split(".")[0]:
            img = cv.imread(file)
            crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            cv.imwrite(os.path.join(pathproposed, file.split("/")[-1]+"_crop.png"), crop_img)
            
    for file in glob("%s/*" % pathridnet):
        if os.path.basename(filename).split(".")[0] == os.path.basename(file).split(".")[0]:
            img = cv.imread(file)
            crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            cv.imwrite(os.path.join(pathridnet, file.split("/")[-1]+"_crop.png"), crop_img)
            
    for file in glob("%s/*" % pathvdnet):
        if os.path.basename(filename).split(".")[0] == os.path.basename(file).split(".")[0]:
            img = cv.imread(file)
            crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            cv.imwrite(os.path.join(pathvdnet, file.split("/")[-1]+"_crop.png"), crop_img)
    
    # crop image and save 
    



