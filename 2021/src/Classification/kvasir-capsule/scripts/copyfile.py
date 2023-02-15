import os 
import glob 
import cv2

names= ["fe5d372e43f94f68_2450","3ada4222967f421d_2415","2fc3db471f9d44c0_1721", "bca26705313a4644_18164"]


dir_path = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/RetinexNet/results/test_results"

for name in names:
    img_path = os.path.join(dir_path, name + ".png")
    # copy to new folder
    os.system("cp %s %s" % (img_path, "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/resultsforPaper/ui/retinex"))

