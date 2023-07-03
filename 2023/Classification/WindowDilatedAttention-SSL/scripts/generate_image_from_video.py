import glob
import os
import argparse
import cv2
# import tqdm as tqdm

parser = argparse.ArgumentParser(
    description="Generate the image from the video.")
parser.add_argument("-d", "--data_dir", type=str)
parser.add_argument("-o", "--output_dir", type=str)


def extract_img(data_dir, dest_dir):
    frame_count = 0
    data_dir = sorted(list(glob.glob("%s/*" % data_dir)),
                      key=lambda x: x.split("/")[-2])
    for video in data_dir:
        video_name = os.path.basename(video)
        video_name = video_name.split(".")[0]
        dest_dir = dest_dir + "/" + video_name
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        cap = cv2.VideoCapture(video)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if ret is False:
                break
            cv2.imwrite(os.path.join(
                dest_dir, str(frame_count) + ".jpg"), frame)
            frame_count += 1
        cap.release()


if __name__ == "__main__":
    args = parser.parse_args()
    data_dir = args.data_dir
    dest_dir = args.output_dir
    extract_img(data_dir, dest_dir)
