# random delete some image in the folder to reduce the number of images to 2000

import os
import random
import argparse

parser = argparse.ArgumentParser(
    description="Randomly cut the images in the folder to reduce the number of images to 2000")

file_dir = os.path.dirname(os.path.realpath(__file__))

parser.add_argument("-s", "--src_dir", type=str,
                    default="/Volumes/Macintosh HD - Data/These/vscode-workspace/Data/Pseudo/Normal clean mucosa/")
parser.add_argument("-f", "--number-of-images", type=int, default=2000)


def reduce_image_number(data_dir, number):
    file_path = os.listdir(data_dir)
    print(len(file_path))
    # suffle the file_path list to get a random list
    file_path = random.sample(file_path, len(file_path))

    for i in range(len(file_path) - number + 1):
        os.remove(os.path.join(data_dir, file_path[i]))


if __name__ == "__main__":
    args = parser.parse_args()
    reduce_image_number(args.src_dir, args.number_of_images)
