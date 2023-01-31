import numpy as np
import os
from tqdm import tqdm
import shutil
import glob

# define a function to read the image from the foler and find the reference image


def readImageDist(distPath, refPath, savePath):
    # create the savePath if it does not exist
    os.makedirs(savePath, exist_ok=True)
    # refPath = os.path.join(refPath, '/*/*/*.jpg')
    # read the image from the folder distPath and find the reference image refPath with the same name and save it to the savePath

    # get all the image names in the distPath
    for root, dirs, files in os.walk(distPath):
        for file in tqdm(files):
            # get the image name
            name = file.split('.')[0]
            found = False

            #  find the reference image refPath subfolder with the same name and save it to the savePath
            if not found:
                for file in glob.glob(refPath):
                    if os.path.basename(file).split('.')[0] == name:
                        print("here====================================================")
                        
                        # get the save image path
                        saveImage = os.path.join(savePath, name + '.jpg')
                        # copy the reference image to the savePath
                        shutil.copy(file, os.path.join(savePath, name + '.jpg'))
                        
                        found = True
                        break
            

if __name__ == "__main__":
    # define the path of the distorted image and the reference image
    refPath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/Ref/*/*/*.jpg'
    distPath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Blur_var/train'
    # define the path to save the reference image
    savePath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Blur_var/train_groundtruth'
    readImageDist(distPath, refPath, savePath)

    distPath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Blur_var/val'
    savePath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Blur_var/val_groundtruth'

    readImageDist(distPath, refPath, savePath)

    distPath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Blur_var/test'
    savePath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Blur_var/test_groundtruth'

    readImageDist(distPath, refPath, savePath)

    distPath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Noise_var/train'
    savePath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Noise_var/train_groundtruth'

    readImageDist(distPath, refPath, savePath)

    distPath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Noise_var/val'
    savePath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Noise_var/val_groundtruth'

    readImageDist(distPath, refPath, savePath)

    distPath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Noise_var/test'
    savePath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Noise_var/test_groundtruth'

    readImageDist(distPath, refPath, savePath)

    distPath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/UI_var/train'
    savePath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/UI_var/train_groundtruth'

    readImageDist(distPath, refPath, savePath)

    distPath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/UI_var/val'
    savePath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/UI_var/val_groundtruth'

    readImageDist(distPath, refPath, savePath)

    distPath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/UI_var/test'
    savePath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/UI_var/test_groundtruth'

    readImageDist(distPath, refPath, savePath)
