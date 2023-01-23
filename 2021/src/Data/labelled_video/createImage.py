import numpy as np
import cv2
# import csv
import os
import glob
from tqdm import tqdm
from calDirection import calDirection
# from functools import reduce
from skimage.util import random_noise

# run the main function
mask_dir = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/forSubTest/videoReadGUI/fps5/mask/"
process_mask = cv2.imread(
    "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/mask.png")
sigmas = [5, 10, 15, 30]
datapath = "/home/nguyentansy/DATA/PhD-work/wceTestPage/src/testVisual/imageExp/ref_images/*/*.jpg"
saved_path = "/home/nguyentansy/DATA/PhD-work/wceTestPage/src/testVisual/imageExp/dist_images/"


def applyui(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.where(hsv[:, :, 2] > 20, mask /
                            255*hsv[:, :, 2], hsv[:, :, 2])
    res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return np.array(res, dtype=np.uint8)


def create_noise(image, sigma, mean=0):
    noise_img = random_noise(image, mode='gaussian',
                             mean=mean,  var=(sigma/255)**2, clip=True)
    return np.array(255*noise_img, dtype='uint8')


mask = cv2.imread(
    "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/mask.png")


def apply_motion_blur(image, size, angle):
    # gaussian filter
    filtered = cv2.GaussianBlur(image, (25, 25), 0)
    dst = np.where(mask == np.array([0, 0, 0]), filtered, image)
    k = np.zeros((size, size), dtype=np.float32)
    k[(size - 1) // 2, :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D(
        (size / 2 - 0.5, size / 2 - 0.5), angle, 1.0), (size, size))
    k = k * (1.0 / np.sum(k))
    out = np.where(mask == np.array([0, 0, 0]),
                   image, cv2.filter2D(dst, -1, k))
    return out


sigmas = [5, 10, 15, 30]
sizes = [5, 10, 15, 25]
sigmas_db = [1, 2, 3, 5]
sizes_ui = [100, 150, 200, 250]
angles = [168]


def apply_defocus_blur(image, sigma):
    dst = cv2.GaussianBlur(
        image, (int(6 * sigma) + 1, int(6 * sigma) + 1), sigma)
    out = np.where(mask == np.array([0, 0, 0]), image, dst)
    return out


if __name__ == "__main__":
    for filename in tqdm(glob.glob(datapath)):
        image = cv2.imread(filename)
        i = 0
        for sigma in sigmas:
            i += 1
            noise_img = create_noise(image, sigma)
            noise_img = np.where(process_mask < 10, image, noise_img)
            # get directory of the datapath
            datapath_dir = os.path.dirname(filename)

            # create directory if not exist
            os.makedirs(os.path.dirname(datapath_dir.replace(
                "ref_images", "dist_images") + "/noise_" + str(i) + ".jpg"), exist_ok=True)

            # save the noise image to the directory
            cv2.imwrite(datapath_dir.replace(
                "ref_images", "dist_images") + "/noise_" + str(i) + ".jpg", noise_img)
        i = 0
        for size in sizes:
            i += 1
            # for angle in angles:
            motion_blur_img = apply_motion_blur(image, size, 45)
            motion_blur_img = np.where(
                process_mask < 10, image, motion_blur_img)

            # get directory of the datapath
            datapath_dir = os.path.dirname(filename)
            print(datapath_dir.replace("ref_images", "dist_images") +
                  "/motion_blur_" + str(size) + ".jpg")

            # create directory if not exist
            os.makedirs(os.path.dirname(datapath_dir.replace(
                "ref_images", "dist_images") + "/motion_blur_" + str(i) + ".jpg"), exist_ok=True)

            # save the noise image to the directory
            cv2.imwrite(datapath_dir.replace(
                "ref_images", "dist_images") + "/motion_blur_" + str(i) + ".jpg", motion_blur_img)

        i = 0
        for sigma in sigmas_db:
            i += 1
            defocus_blur_img = apply_defocus_blur(image, sigma)
            defocus_blur_img = np.where(
                process_mask < 10, image, defocus_blur_img)
            # get directory of the datapath
            datapath_dir = os.path.dirname(filename)
            print(datapath_dir.replace("ref_images", "dist_images") +
                  "/defocus_blur_" + str(sigma) + ".jpg")

            # create directory if not exist
            os.makedirs(os.path.dirname(datapath_dir.replace(
                "ref_images", "dist_images") + "/defocus_blur_" + str(i) + ".jpg"), exist_ok=True)

            # save the noise image to the directory
            cv2.imwrite(datapath_dir.replace(
                "ref_images", "dist_images") + "/defocus_blur_" + str(i) + ".jpg", defocus_blur_img)

        i = 0
        for size in tqdm(sizes_ui):
            i += 1
            for angle in angles:
                mask = cv2.imread(mask_dir+str(size)+"_" +
                                  str(angle)+".png", cv2.IMREAD_GRAYSCALE)
                for file in tqdm(glob.glob(datapath)):
                    frame = cv2.imread(file)
                    noise_img = applyui(frame, mask=mask)
                    finalnoise = np.where(
                        process_mask < 10, frame, noise_img)
                    print(datapath_dir.replace("ref_images", "dist_images") +
                          "/UI_" + str(i)  + ".jpg")

                    # save the noise image to the directory
                    cv2.imwrite(datapath_dir.replace("ref_images", "dist_images") +
                                "/UI_" + str(i)  + ".jpg", finalnoise)
