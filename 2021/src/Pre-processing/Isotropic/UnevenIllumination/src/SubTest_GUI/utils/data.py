# src code to load the videos

# import the necessary packages
import os
import random

def load_data(src_data_path):
    # show notification
    print("Loading data...")
    src_reference_video_path =  src_data_path + '/ref_videos/'
    src_distorted_video_path = src_data_path + '/dist_videos/'

    # load reference videos from src_reference_video_path 
    reference_video_paths = [os.path.join(src_reference_video_path, f) for f in os.listdir(src_reference_video_path)]

    # load every distorted video from src_distorted_video_path using os.walk
    distorted_video_paths = []
    distortion_types = []
    for root, dirs, files in os.walk(src_distorted_video_path):
        for file in files:
            if file.endswith(".mp4"):
                distorted_video_paths.append(os.path.join(root, file))
                distortion_types.append(os.path.basename(os.path.dirname(root)))
    
    # show notification
    print("Data loaded successfully!")
    print('Number of distorted videos: ',len(distorted_video_paths))
    print('Number of reference videos: ',len(reference_video_paths))

    # zip the video paths and distortion types
    data = list(zip(distorted_video_paths, distortion_types))

    #  suffle data
    random.shuffle(data)

    # unzip the data
    distorted_video_paths, distortion_types = zip(*data)

    return reference_video_paths, distorted_video_paths, distortion_types

# get reference video in the reference_videos path where the reference videos have the same name as the distorted videos
def get_reference_video(distorted_video_path,reference_video_paths):
    # get the name of the distorted video
    distorted_video_name = os.path.basename(distorted_video_path)
    # find the reference video with the same name as the distorted video in the reference_video_paths
    for reference_video_path in reference_video_paths:
        if distorted_video_name == os.path.basename(reference_video_path):
            return reference_video_path
    return reference_video_path

if __name__ == '__main__':
    # load the reference videos and distorted videos
    reference_video_paths, distorted_video_paths = load_data('./videos')