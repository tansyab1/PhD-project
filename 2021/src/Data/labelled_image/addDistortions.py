
import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.util import random_noise
# from addNoise import create_noise
# from addBlur import apply_defocus_blur, apply_motion_blur
# from createUI import applyui

def create_noise(image, sigma, mean=0):
    noise_img = random_noise(image, mode='gaussian',
                             mean=mean,  var=(sigma/255)**2, clip=True)
    return np.array(255*noise_img, dtype='uint8')


def apply_motion_blur(image, size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[(size - 1) // 2, :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D(
        (size / 2 - 0.5, size / 2 - 0.5), angle, 1.0), (size, size))
    k = k * (1.0 / np.sum(k))
    dst = cv2.filter2D(image, -1, k)
    out = np.where(mask == np.array([0, 0, 0]), image, dst)
    return out


def apply_defocus_blur(image, sigma):
    dst = cv2.GaussianBlur(image, (int(6 * sigma) + 1, int(6 * sigma) + 1), sigma)
    out = np.where(mask == np.array([0, 0, 0]), image, dst)
    return out

def applyui(image, mask):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.where(hsv[:,:,2]>20, mask/255*hsv[:,:,2],hsv[:,:,2])
    res = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return np.array(res,dtype=np.uint8)

names = ["Ampulla of vater",
         "Angiectasia",
         "Blood - fresh",
         "Blood - hematin",
         "Erosion",
         "Erythema",
         "Foreign body",
         "Ileocecal valve",
         "Lymphangiectasia",
         "Normal clean mucosa",
         "Polyp",
         "Pylorus",
         "Reduced mucosal view",
         "Ulcer"]

sigma_Noise = [5, 10, 15, 20, 30, 40]
sigma_DefocusBlur = [0.75, 1, 2, 3, 5]
angle_MotionBlurs = [0, 45, 90, 135]
size_MotionBlurs = [5, 10, 15, 25]
size_UIs = [100, 150, 200, 250]
angle_UIs = [112,168,224]

mask = cv2.imread("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/distorted_images/mask.png")
mask_dir = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/distorted_images/UI/mask/"
save_dir = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/distorted_images/"
input_dir = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ref/"

# print("Start adding noise")
# # apply Noise to input images
# for sig_Noise in tqdm(sigma_Noise):
#     for name in names:
#         # check if the folder exists
#         if not os.path.exists(save_dir+"Noise/"+str(sig_Noise)+"/"+name+"/"):
#             os.makedirs(save_dir+"Noise/"+str(sig_Noise)+"/"+name+"/")
#         for file in tqdm(glob.glob(input_dir+name+"/*.jpg")):
#             image = cv2.imread(file)
#             noise_img = create_noise(image, sig_Noise)
#             finalnoise= np.where(mask <10, image, noise_img)
#             cv2.imwrite(save_dir+"Noise/"+str(sig_Noise)+"/"+name+"/"+os.path.basename(file), finalnoise)


# print("Start adding blur")
# # apply Defocus Blur to input images
# for sig_DefocusBlur in tqdm(sigma_DefocusBlur):
#     for name in names:
#         # check if the folder exists
#         if not os.path.exists(save_dir+"DefocusBlur/"+str(sig_DefocusBlur)+"/"+name+"/"):
#             os.makedirs(save_dir+"DefocusBlur/"+str(sig_DefocusBlur)+"/"+name+"/")
#         for file in tqdm(glob.glob(input_dir+name+"/*.jpg")):
#             image = cv2.imread(file)
#             blur_img = apply_defocus_blur(image, sig_DefocusBlur)
#             finalblur= np.where(mask <10, image, blur_img)
#             cv2.imwrite(save_dir+"DefocusBlur/"+str(sig_DefocusBlur)+"/"+name+"/"+os.path.basename(file), finalblur)


# # apply Motion Blur to input images
# for angle_MotionBlur in tqdm(angle_MotionBlurs):
#     for size_MotionBlur in tqdm(size_MotionBlurs):
#         for name in names:
#             # check if the folder exists
#             if not os.path.exists(save_dir+"MotionBlur/"+str(angle_MotionBlur)+"/"+str(size_MotionBlur)+"/"+name+"/"):
#                 os.makedirs(save_dir+"MotionBlur/"+str(angle_MotionBlur)+"/"+str(size_MotionBlur)+"/"+name+"/")
#             for file in tqdm(glob.glob(input_dir+name+"/*.jpg")):
#                 image = cv2.imread(file)
#                 blur_img = apply_motion_blur(image, size_MotionBlur, angle_MotionBlur)
#                 finalblur= np.where(mask <10, image, blur_img)
#                 cv2.imwrite(save_dir+"MotionBlur/"+str(angle_MotionBlur)+"/"+str(size_MotionBlur)+"/"+name+"/"+os.path.basename(file), finalblur)

print("Start adding UI")
# apply UI to input images
for size_UI in tqdm(size_UIs):
    for angle_UI in tqdm(angle_UIs):
        ui_mask = cv2.imread(mask_dir+str(size_UI)+"_"+str(angle_UI)+".png", cv2.IMREAD_GRAYSCALE)
        for name in names:
            # check if the folder exists
            if not os.path.exists(save_dir+"UI/"+str(size_UI)+"/"+str(angle_UI)+"/"+name+"/"):
                os.makedirs(save_dir+"UI/"+str(size_UI)+"/"+str(angle_UI)+"/"+name+"/")
            for file in tqdm(glob.glob(input_dir+name+"/*.jpg")):
                image = cv2.imread(file)
                ui_img = applyui(image, ui_mask)
                finalui= np.where(mask <10, image, ui_img)
                cv2.imwrite(save_dir+"UI/"+str(size_UI)+"/"+str(angle_UI)+"/"+name+"/"+os.path.basename(file), finalui)