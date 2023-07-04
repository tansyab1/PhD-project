import glob
import os
import argparse
import cv2
from tqdm import tqdm as tqdm

parser = argparse.ArgumentParser(
    description="Generate the image from the video.")
parser.add_argument("-d", "--data_dir", type=str, default="/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/unlabelled_videos")
parser.add_argument("-o", "--output_dir", type=str, default="/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/unlabelled_videos_images")


def extract_img(data_dir, dest_dir):
    frame_count = 0
    data_dir = sorted(list(glob.glob("%s/*" % data_dir)),
                      key=lambda x: x.split("/")[-2])
    for video in tqdm(data_dir):
        video_name = os.path.basename(video)
        video_name = video_name.split(".")[0]
        cap = cv2.VideoCapture(video)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if ret is False:
                break
            cv2.imwrite(os.path.join(
                dest_dir, video_name + "_%d.jpg" % frame_count), frame)
            frame_count += 1
        cap.release()


if __name__ == "__main__":
    args = parser.parse_args()
    data_dir = args.data_dir
    dest_dir = args.output_dir
    extract_img(data_dir, dest_dir)
