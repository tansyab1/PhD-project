# import the libraries moviepy and tqdm
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import os
import glob


# define the function to read video, add noise and save the video
def addNoise(video_path, save_path):
    # check if the video exists
    if not os.path.exists(video_path):
        print('Video not found!')
        return
    # create the directory to save the video
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    # read the video
    video = VideoFileClip(video_path)
    # add noise to the video
    noise_video = video.fx(vfx.gaussian_noise, var=0.1)
    # save the video
    noise_video.write_videofile(save_path)


# define the main function
def main():
    video_path = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest'
    # save_path = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest'
    # read all the videos in the folder and subfolders
    for root, dirs, files in os.walk(video_path):
        for file in tqdm(files):
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                print(video_path)
                # change the video paths inside the video_path to save_path
                save_path = video_path.replace('forSubTest', 'forSubTest_noise')
                addNoise(video_path, save_path)
                
