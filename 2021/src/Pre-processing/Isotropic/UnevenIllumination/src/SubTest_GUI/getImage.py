# python file to read video and get the image at the second 8

import cv2
import numpy as np
import os
import sys
import random
from tqdm import tqdm

# function to get the image at the second 8
def getImage(video_path):
    # check if the video exists
    if not os.path.exists(video_path):
        print('Video not found!')
        return
    # read the video
    cap = cv2.VideoCapture(video_path)
    # get the number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get the fps
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # get the width
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # get the height
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # get the duration of the video
    duration = frame_count/fps
    # get the frame at the second 8
    cap.set(cv2.CAP_PROP_POS_FRAMES, 8*fps)
    # read the frame
    ret, frame = cap.read()
    # check if the frame is read successfully
    if not ret:
        print('Error in reading the frame!')
        return
    # save the image with name is the name of the containing folder
    cv2.imwrite(os.path.join(os.path.dirname(video_path), os.path.basename(os.path.dirname(video_path))) + '.jpg', frame)

    # release the video
    cap.release()

# link to the folder contains the noise videos
noise_dir = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest/forImage/noise'
# link to the folder contains the motion blur videos
motion_dir = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest/forImage/mblur'
# link to the folder contains the defocus blur videos
defocus_dir = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest/forImage/dblur'
# link to the folder contains the ui videos
ui_dir = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest/forImage/ui'

# get all videos in the noise, motion blur, defocus blur and ui folders and subfolders
noise_videos = [os.path.join(dp, f) for dp, dn, filenames in os.walk(noise_dir) for f in filenames if os.path.splitext(f)[1] == '.mp4']
motion_videos = [os.path.join(dp, f) for dp, dn, filenames in os.walk(motion_dir) for f in filenames if os.path.splitext(f)[1] == '.mp4']
defocus_videos = [os.path.join(dp, f) for dp, dn, filenames in os.walk(defocus_dir) for f in filenames if os.path.splitext(f)[1] == '.mp4']
ui_videos = [os.path.join(dp, f) for dp, dn, filenames in os.walk(ui_dir) for f in filenames if os.path.splitext(f)[1] == '.mp4']

# get the image at the second 8 for all videos  
for video in tqdm(noise_videos):
    getImage(video)
for video in tqdm(motion_videos):
    getImage(video)
for video in tqdm(defocus_videos):
    getImage(video)
for video in tqdm(ui_videos):
    getImage(video)

