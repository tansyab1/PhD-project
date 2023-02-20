import os 
import glob 
import cv2

# names= ["fe5d372e43f94f68_2450","3ada4222967f421d_2415","2fc3db471f9d44c0_1721", "bca26705313a4644_18164"]

# names= ["dc221ccc65d34010_16396", "ed02f27ef36f483e_13897", "6cb700585c4f4070_12922", "fb86bc87d3874cd7_5293"]

names = ["3c8d5f0b90d7475d_4804", "c7084b3556e34619_29853", "0728084c8da942d9_29988", "8885668afb844852_3813"]




dir_path = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Blur_var/test/input"

for name in names:
    img_path = os.path.join(dir_path, name + ".jpg")
    # copy to new folder
    os.system("cp %s %s" % (img_path, "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/resultsforPaper/ref/blur/"))

