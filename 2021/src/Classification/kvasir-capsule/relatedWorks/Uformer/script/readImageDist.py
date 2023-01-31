import numpy as np
import os
import tqdm as tqdm
import glob

# define a function to read the image from the foler and find the reference image


def readImageDist(distPath, refPath, savePath):
    # create the savePath if it does not exist
    os.makedirs(savePath, exist_ok=True)
    refPath = os.path.join(refPath, '/*/*/*.jpg')
    # read the image from the folder distPath and find the reference image refPath with the same name and save it to the savePath

    # get all the image names in the distPath
    for root, dirs, files in os.walk(distPath):
        for file in tqdm.tqdm(files):
            # get the image name
            name = file.split('.')[0]

            #  find the reference image refPath subfolder with the same name and save it to the savePath

            for file in tqdm(glob.glob(refPath)):

                if file == name + '.jpg':
                    # get the reference image path from the refPath
                    refImage = os.path.join(root, dirs)
                    refImage = os.path.join(refImage, file)

                    # get the save image path
                    saveImage = os.path.join(savePath, name + '.jpg')
                    # copy the reference image to the savePath
                    os.system('cp ' + refImage + ' ' + saveImage)
                    break
                break


if __name__ == "__main__":
    # define the path of the distorted image and the reference image
    refPath = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/Ref'
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
