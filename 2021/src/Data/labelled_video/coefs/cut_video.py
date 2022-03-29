from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import glob
from tqdm import tqdm
import os
# import cv2


visualize_path = "src/Pre-processing/Isotropic/KvasirCapsule/coefs/visualization/"

for file in tqdm(glob.glob("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/*.mp4")):
    oriname = os.path.basename(file)
    fig,ax = plt.subplots()
    visual = plt.imread(visualize_path+oriname+".png")
    ax.imshow(visual)
    plt.show()
    start = input("Enter start time: ")
    end = input("Enter end time: ")
    dst_name= "ref_"+os.path.splitext(oriname)[0]+".mp4"
    ffmpeg_extract_subclip(file, int(start), int(end)-1, targetname="src/Pre-processing/Isotropic/KvasirCapsule/coefs/target_videos/"+dst_name)