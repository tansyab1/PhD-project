
# import from addNoise.py
import numpy as np
import cv2
# import csv
import os
import glob
from tqdm import tqdm
from functools import reduce
from skimage.util import random_noise

# import all functions from addNoise.py
from addNoise import create_noise, addNoise
# import all functions from addUI.py
from addUI import addUI
# import all functions from addBlur.py
from addBlur import addBlur
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

datapath = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/\
    labelled_videos/process/forSubTest/videoReadGUI/fps5/*/*/*.mp4"
datapath_ui = "2021/src/Data/labelled_video/ref_4aebc5cb2d4847aa.mp4"
# savepath = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest/videoReadGUI/fps5/cut/"

# define the function to cut the video from second 15 to second 45


# def cutVideo():
#     # read all videos in the folder
#     # for video in tqdm(glob.glob(datapath)):
#     #     savepath = video.replace("fps5", "cut")
#     #     # create path if not exist

#     #     os.makedirs(os.path.dirname(savepath), exist_ok=True)

#     #     # cut video from second 15 to second 45

#     #     ffmpeg_extract_subclip(
#     #         video, 15, 45, targetname=savepath)

#     for video in tqdm(glob.glob(datapath_ui)):
#         savepath = video.replace("fps5", "cut")
#         # create path if not exist

#         os.makedirs(os.path.dirname(savepath), exist_ok=True)

#         # cut video from second 15 to second 45

#         ffmpeg_extract_subclip(
#             video, 15, 45, targetname=savepath)

    # run the main function
if __name__ == "__main__":
    # cutVideo()
    # addNoise()
    # addUI()
    addBlur()
