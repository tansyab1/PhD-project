import glob
import csv
import os
from pathlib import Path
import cv2
import numpy as np
# from functools import reduce
from skimage.util import random_noise
from tqdm import tqdm
import matplotlib.pyplot as plt
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
names = []
times = []
save_folder = 'D:/Datasets/kvasircapsule/labelled_videos/ref/'

print(os.path.exists("D:/Datasets"))
for file in tqdm(glob.glob('D:/Datasets/kvasircapsule/labelled_videos/*.mp4')):
    image = cv2.imread('D:/Datasets/kvasircapsule/labelled_videos/visualization/'+os.path.basename(file) + '.png')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    print('Enter time:')
    time = input()
    names.append(os.path.basename(file))
    times.append(time)
    ffmpeg_extract_subclip(file, int(time), int(time)+300, targetname=save_folder+"ref_"+os.path.basename(file))

with open('annotation.csv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(names, times))