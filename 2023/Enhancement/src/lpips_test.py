"""calculate LPIPS for each image pair in the dataset and save the results in a mat file
    Args:
        ref_path: path to the reference images
        dis_path: path to the distorted images
        save_path: path to save the results
    Returns:
        None
    """

import os
import torch
import numpy as np
import scipy.io as sio
import lpips
from tqdm import tqdm
from torchvision.io import read_image

loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
def calculateLPIPS(ref_path, dis_path, save_path):


    ref_list = os.listdir(ref_path)
    dis_list = os.listdir(dis_path)
    ref_list.sort()
    dis_list.sort()

    num = len(ref_list)
    lpips_list = []
    for i in tqdm(range(num)):
        # check is the image or not
        if dis_list[i].lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            # check if can read_image or not
            try:
                dis_img = read_image(dis_path + dis_list[i]).cuda()
                # find the image with the same name in the reference image folder and calculate the LPIPS
                base = os.path.basename(dis_list[i])
                # base ref is the same as base dis with extension .jpg
                base_ref = base.split(".")[0] + ".jpg"
                if base_ref.endswith("_DUAL_g0.6_l0.15.jpg"):
                    base_ref = base_ref.replace("_DUAL_g0.6_l0.15.jpg", ".jpg")
                ref_img = read_image(ref_path + base_ref).cuda()
                lpips_list.append(loss_fn_vgg(ref_img, dis_img).item())
            except:
                print("Cannot read image: " + dis_list[i])
        else:
            print("Not an image file: " + dis_list[i])

    #     dis_img = read_image(dis_path + dis_list[i]).cuda()
    #     # find the image with the same name in the reference image folder and calculate the LPIPS
    #     base = os.path.basename(dis_list[i])
    #     ref_img = read_image(ref_path + base).cuda()
    #     lpips_list.append(loss_fn_vgg(ref_img, dis_img).item())
        # show the average LPIPS
    print("value of LPIPS of " + save_path + " is: " + str(np.mean(lpips_list)))
    # save the results in a mat file
    save_path = str(np.mean(lpips_list)) + "_" + save_path
    sio.savemat(save_path, {'lpips': lpips_list})
    # empty cuda cache to avoid memory leak


if __name__ == '__main__':
    dis_path = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/forNoise/BM3D/"
    ref_path = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Noise_var/test/groundtruth/"
    # save_path = 'lpips_BM3D.mat'
    # calculateLPIPS(ref_path, dis_path, save_path)
    
    dis_paths = ["/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/forNoise/MPRNet/",
                 "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/forNoise/VDNet/",
                 "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/forNoise/Uformer/"]
    for dis_path in dis_paths:
        save_path = "noise_lipis_" + dis_path.split("/")[-2] + ".mat"
        # print(save_path)
        calculateLPIPS(ref_path, dis_path, save_path)
    
    resultpath = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/"
        
    dis_paths = [resultpath + "forBlur/DBGAN/",
                 resultpath + "forBlur/DeblurGANv2/",
                 resultpath + "forBlur/DMPHN/",
                 resultpath + "forBlur/Uformer/"]
    for dis_path in dis_paths:
        save_path = "blur_lipis_" + dis_path.split("/")[-2] + ".mat"
        # print(save_path)
        calculateLPIPS(ref_path, dis_path, save_path)
        
    dis_paths = [resultpath + "forUI/RetinexNet/",
                 resultpath + "forUI/EnlightenGAN/",
                 resultpath + "forUI/Uformer/",
                 resultpath + "forUI/LIME/"]
    
    for dis_path in dis_paths:
        save_path = "ui_lipis_" + dis_path.split("/")[-2] + ".mat"
        # print(save_path)
        calculateLPIPS(ref_path, dis_path, save_path)