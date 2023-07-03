# file to crop the center of the image and save it to a new folder with the same name as the original folder

import os
import argparse
import cv2
import glob


parser = argparse.ArgumentParser(
    description="Generate a annotation file from a class file strucutre.")

parser.add_argument("-d", "--data_dir", type=str)


def crop_center(data_dir, size, out_dir):
    file_path = sorted(list(glob.glob("%s/*" % data_dir)),
                       key=lambda x: x.split("/")[-2])
    for file in file_path:
        # print(file)
        image = cv2.imread(file)
        x = image.shape[0]
        y = image.shape[1]
        x_start = x // 2 - size // 2
        x_end = x // 2 + size // 2
        y_start = y // 2 - size // 2
        y_end = y // 2 + size // 2
        image = image[x_start:x_end, y_start:y_end]
        # resize the image to 336x336
        image = cv2.resize(image, (336, 336))
        # save the image to the out_dir
        cv2.imwrite(os.path.join(out_dir, os.path.basename(file)), image)


if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.data_dir.replace("polyps", "polyps_crop")

    files_dir = os.listdir(data_dir)
    # minsize = 500
    # for file in files_dir:
    #     img = cv2.imread(os.path.join(data_dir, file))
    #     if img is not None:
    #         minsize = min(minsize, min(img.shape[:2]))
    #         if min(img.shape[:2]) < 420:
    #             # remove the image
    #             os.remove(os.path.join(data_dir, file))
    # # print(minsize)

    crop_center(data_dir, 420, out_dir)
