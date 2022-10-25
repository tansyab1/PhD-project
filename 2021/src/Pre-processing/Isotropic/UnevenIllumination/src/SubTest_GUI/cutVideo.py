# python file to read video and cut first 10 seconds

import cv2
import numpy as np
import os
import sys
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm

# define the function to read video and cut first 10 seconds
def cutVideo(video_path, save_path):
    # check if the video exists
    if not os.path.exists(video_path):
        print('Video not found!')
        return
    # create the directory to save the video
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    ffmpeg_extract_subclip(video_path, 10, 20, targetname=save_path)

# define the main function
def main():
    video_path = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/ref'
    video_ui_path = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/UI'
    video_defocus_path = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/Blur/Defocus'
    video_motion_blur_path = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/Blur/Motion_bs'
    video_noise_path = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/Noise'
    # save_path = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest'
    # read all the videos in the folder and subfolders
    for root, dirs, files in os.walk(video_path):
        for file in tqdm(files):
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                print(video_path)
                # change the video paths inside the video_path to save_path
                save_path = video_path.replace('labelled_videos_process', 'forSubTest')
                cutVideo(video_path, save_path)
    for root, dirs, files in os.walk(video_ui_path):
        for file in tqdm(files):
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                print(video_path)
                # change the video paths inside the video_path to save_path
                save_path = video_path.replace('labelled_videos_process', 'forSubTest')
                cutVideo(video_path, save_path)
    for root, dirs, files in os.walk(video_defocus_path):
        for file in tqdm(files):
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                print(video_path)
                # change the video paths inside the video_path to save_path
                save_path = video_path.replace('labelled_videos_process', 'forSubTest')
                cutVideo(video_path, save_path)
    for root, dirs, files in os.walk(video_motion_blur_path):
        for file in tqdm(files):
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                print(video_path)
                # change the video paths inside the video_path to save_path
                save_path = video_path.replace('labelled_videos_process', 'forSubTest')
                cutVideo(video_path, save_path)
    for root, dirs, files in os.walk(video_noise_path):
        for file in tqdm(files):
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                print(video_path)
                # change the video paths inside the video_path to save_path
                save_path = video_path.replace('labelled_videos_process', 'forSubTest')
                cutVideo(video_path, save_path)
                

# call the main function
if __name__ == '__main__':
    main()