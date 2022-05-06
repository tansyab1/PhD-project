
import numpy as np
import cv2
import csv
import os
import glob
from tqdm import tqdm
from functools import reduce
from checkNoise import estimateStandardDeviation
from checkBlur import getGlobalBlur
from checkUI import getIHED

names = ["Ampulla of vater", 
         "Angiectasia",
         "Blood - fresh",
         "Blood - hematin",
         "Erosion",
         "Erythema",
         "Foreign body",
         "Ileocecal valve",
         "Lymphangiectasia",
         "Normal clean mucosa",
         "Polyp",
         "Pylorus",
         "Reduced mucosal view",
         "Ulcer"]
save_folder ='/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/'\
    'labelled_images/process/labelled_images/visualization/new_index/'
for name in names:
    ui = []
    noise = []
    blur = []
    for file in tqdm(glob.glob("/home/nguyentansy/DATA/PhD-work/Datasets/"\
        "kvasir_capsule/labelled_images/"+name+"/*.jpg")):

        frame =cv2.imread(file)
        # Display the resulting frame
        ui_coefficient = getIHED(frame)
        noise_coefficient = estimateStandardDeviation(frame)
        blur_coefficient = getGlobalBlur(frame)

        ui.append(ui_coefficient[0][0])
        noise.append(noise_coefficient)
        blur.append(blur_coefficient)


    with open(save_folder + name+'.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(ui, noise, blur))
