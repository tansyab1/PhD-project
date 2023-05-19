# source to save all images from video

import os
import cv2


def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    # get video name
    video_name = pathIn.split('/')[-1].split('.')[0]
    success, image = vidcap.read()
    while success:
        cv2.imwrite(pathOut + video_name + "_frame%d.jpg" % count, image)
        count += 1
        success, image = vidcap.read()
    print("Done")


# read all videos in folder
pathIn = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/main/KvasirCapsuleIQA/final/'
pathOut = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/main/KvasirCapsuleIQA/final/imgs/'
for video in os.listdir(pathIn):
    if video.endswith(".mp4"):
        extractImages(pathIn + video, pathOut)
