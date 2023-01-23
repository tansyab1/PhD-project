# python fie to read and change fps of the video

import cv2
import os
import tqdm as tqdm
import glob
import numpy as np


#  define a function to select randomly one video from the folder 

def selectUnique(dir):
    # get all .mp4 files in the folder
    
    list_videos = glob.glob(dir + "/*/*.mp4")
    # saved_dir = dir.replace("UI", "selected_UI")
    
    # suffle the list
    list_videos_basenames = [os.path.basename(x) for x in list_videos]
    
    # get the list with unique videos
    
    list_videos_unique = list(set(list_videos_basenames))
    
    # for each unique video, select one randomly in the list_videos with the same name and save it to the new folder
    
    for video in list_videos_unique:
        # get the list of all videos with the same name
        list_videos_same_name = [x for x in list_videos if video in x]
        # select one randomly
        selected_video = np.random.choice(list_videos_same_name)
        saved_dir=selected_video.replace("UI", "selected_UI")
        # save it to the new folder with the same name
        os.system('cp {} {}'.format(selected_video, saved_dir))