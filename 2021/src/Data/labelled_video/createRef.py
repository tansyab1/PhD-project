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
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import ffmpeg
names = []
times = []
pathos = []
save_folder = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/main/final/'

# print(os.path.exists("D:/Datasets"))
for file in tqdm(glob.glob('/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/main/original/*.mp4')):
    # image = cv2.imread('D:/Datasets/kvasircapsule/labelled_videos/visualization/'+os.path.basename(file) + '.png')
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()
    basename = os.path.basename(file).split('_')[0]
    time1 = os.path.basename(file).split('_')[1]
    time2 = os.path.basename(file).split('_')[2]
    patho = os.path.basename(file).split('_')[3]
    
    savename = basename +"_" +patho

    # print('Enter time:')
    # time = input()
    names.append(basename)
    pathos.append(patho)
    times.append(time1)
    
    #  trim video using ffmpeg from time1 to time2
    input_stream = ffmpeg.input(file)
    # set up ffmpeg to have same codec, fps, etc. as original video
    
    
    
    output_stream = ffmpeg.output(input_stream, save_folder+savename, ss=int(time1), to=int(time1)+10)
    ffmpeg.run(output_stream)

with open(save_folder+'annotation.csv', 'w', newline='') as csvfile:
    fieldnames = ['video_name', 'pathology', 'time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(names)):
        writer.writerow({'video_name': names[i], 'pathology': pathos[i], 'time': times[i]})