import numpy as np
import cv2
import csv
import os
import glob
from tqdm import tqdm
from functools import reduce
from checkNoise import estimateStandardDeviation
from checkBlur import getVarianofLaplace
from checkUI import getIHED
import pandas as pd

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
save_folder ='/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ref/'
df_data = pd.read_csv("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/visualization/threshold.csv", delimiter=',', header=None)

thresholds = df_data.values[:, 1]
names = df_data.values[:, 0]
for name in names:
    f_data = pd.read_csv("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/"+name+".csv", delimiter='\t', header=None)

    # estimated coefficients of different models
    f_data.columns = ['ui', 'noise', 'blur']

    data_ui = max(f_data.values[:, 0])
    data_noise = max(f_data.values[:, 1])
    data_blur = max(1/f_data.values[:, 2])
    std_ui = np.std(f_data.values[:, 0])
    std_noise = np.std(f_data.values[:, 1])
    std_blur = np.std(1/f_data.values[:, 2])
    ui = []
    noise = []
    blur = []
    for file in tqdm(glob.glob("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/"+name+"/*.jpg")):
        if not os.path.exists(save_folder+name+"/"):
            os.makedirs(save_folder+name+"/")
        frame =cv2.imread(file)
        # Display the resulting frame
        ui_coefficient = getIHED(frame)
        noise_coefficient = estimateStandardDeviation(frame)
        blur_coefficient = getVarianofLaplace(frame)

        over_coef= std_blur/(data_blur*blur_coefficient)+std_ui*ui_coefficient[0][0]/data_ui+std_noise*noise_coefficient/data_noise
        if over_coef < thresholds[np.where(names == name)[0][0]]:
            cv2.imwrite(save_folder+name+"/"+os.path.basename(file), frame)


    # with open(save_folder + name+'.csv', 'w') as f:
    #     writer = csv.writer(f, delimiter='\t')
    #     writer.writerows(zip(ui, noise, blur))
