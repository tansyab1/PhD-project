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
save_folder = 'D:/Datasets/kvasircapsule/labelled_videos/ref/test/'
image = cv2.imread('D:/Datasets/kvasircapsule/labelled_videos/visualization/3c8d5f0b90d7475d.mp4.png')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
file = 'D:/Datasets/kvasircapsule/labelled_videos/original/3c8d5f0b90d7475d.mp4'
print('Enter time:')
time = input()
names.append(os.path.basename('D:/Datasets/kvasircapsule/labelled_videos/original/3c8d5f0b90d7475d.mp4'))
times.append(time)
ffmpeg_extract_subclip(file, int(time), int(time)+10, targetname=save_folder+"ref_"+os.path.basename(file))
