import numpy as np
import os
import tqdm

# define a function to read the image from the foler and find the reference image


def readImageDist(distPath, refPath, savePath):
    # read the image from the folder distPath and find the reference image refPath with the same name and save it to the savePath

    # get all the image names in the distPath
    for root, dirs, files in os.walk(distPath):
        for file in tqdm.tqdm(files):
            # get the image name
            name = file.split('.')[0]

            #  find the reference image refPath with the same name and save it to the savePath

            for root, dirs, files in os.walk(refPath):
                for file in files:
                    if file == name + '.jpg':
                        # get the reference image path
                        refImage = os.path.join(refPath, file)
                        # get the save image path
                        saveImage = os.path.join(savePath, name + '.jpg')
                        # copy the reference image to the savePath
                        os.system('cp ' + refImage + ' ' + saveImage)
                        
if __name__ == "__main__":
    # define the path of the distorted image and the reference image
    refPath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/distorted_images/Blur_var'
    distPath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Blur_var/train'
    # define the path to save the reference image
    savePath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Blur_var/train_groundtruth'
    readImageDist(distPath, refPath, savePath)
