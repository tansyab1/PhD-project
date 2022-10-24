#  python file to read and select video randomly

import cv2
import numpy as np
import os
import sys
import random
from tqdm import tqdm

# save link to variable
ref_dir = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/videoReadGUI/ref'
selected_dir = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/videoReadGUI/select20'
Defocus_dir = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest/Blur/Defocus'
Noise_dir = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest/Noise'
Motion_Blur_dir = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest/Blur/Motion_bs'
UI_dir = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest/UI'

# path for selected videos of the Noise folder
selected_Noise_dir = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/videoReadGUI/Noise'
# path for selected videos of the Motion Blur folder
selected_Motion_Blur_dir = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/videoReadGUI/Blur/Motion_Blur'
# path for selected videos of the Defocus folder
selected_Defocus_dir = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/videoReadGUI/Blur/Defocus'
# path for selected videos of the UI folder
selected_UI_dir = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/videoReadGUI/UI'


# define the function to randomly select 20 videos from the ref folder and copy to the test folder  
def select20Video(ref_dir):
    # check if the video exists
    if not os.path.exists(ref_dir):
        print('Video not found!')
        return
    # read all the videos in the folder and subfolders
    for root, dirs, files in os.walk(ref_dir):
        # randomly select 20 videos
        random.shuffle(files)
        for file in tqdm(files[:20]):
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                print(video_path)
                # change the video paths inside the video_path to save_path
                save_path = video_path.replace('videoReadGUI/ref', 'videoReadGUI/select20')
                # create the directory to save the video
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                # copy the video to the test folder
                os.system('cp {} {}'.format(video_path, save_path))

# define the function to find the video matches the name of the selected video name
def findVideo(given_name, dir):
    # check if the video exists
    if not os.path.exists(dir):
        print('Video not found!')
        return
    
    # list contains the directory of the video matches the name
    list = []
    # read all the videos in the folder and subfolders
    for root, dirs, files in os.walk(dir):
        for file in tqdm(files):
            if file.endswith('.mp4'):
                if file == given_name:
                    video_path = os.path.join(root, file)
                    list.append(video_path)
    return list

# define the function to read the selected videos and find the video with the same name in other folders
def readVideo(selected_dir, distorted_dir, dest_dir):
    # distorted_dir has different subfolders which is the different levels of distortion
    # check if the video exists
    if not os.path.exists(selected_dir):
        print('Video not found!')
        return
    # all folders inside the distorted_dir
    subfolders = os.listdir(distorted_dir)
    #  for each subfolder, find the video with the same name as the selected video
    for subfolder in subfolders:
        # list contains the directory of the video matches the name
        list = []
        # read all the videos in the folder and subfolders
        for root, dirs, files in os.walk(selected_dir):
            for file in tqdm(files):
                if file.endswith('.mp4'):
                    # find the video with the same name as the selected video
                    list = findVideo(file, os.path.join(distorted_dir, subfolder))
                    #  select just one path in list randomly
                    random.shuffle(list)
                    #  save the video to the dest_dir
                    for video_path in list[:1]:
                        # change the video paths inside the video_path to save_path
                        save_path = video_path.replace('forSubTest', 'videoReadGUI')
                        # create the directory to save the video
                        if not os.path.exists(os.path.dirname(save_path)):
                            os.makedirs(os.path.dirname(save_path))
                        # copy the video to the test folder
                        os.system('cp {} {}'.format(video_path, save_path))

                    


#  define the main function
def main():
    #  select 20 videos from the ref folder and copy to the test folder
    select20Video(ref_dir)
    readVideo(selected_dir, Noise_dir, selected_Noise_dir)
    readVideo(selected_dir, Motion_Blur_dir, selected_Motion_Blur_dir)
    readVideo(selected_dir, Defocus_dir, selected_Defocus_dir)
    readVideo(selected_dir, UI_dir, selected_UI_dir)


# call the main function
if __name__ == '__main__':
    main()