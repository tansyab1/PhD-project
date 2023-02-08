"""
## CycleISP: Real Image Restoration Via Improved Data Synthesis
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## CVPR 2020
## https://arxiv.org/abs/2003.07761
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2

import scipy.io as sio
from networks.denoising_rgb import DenoiseNet
from dataloaders.data_rgb import get_test_data
import utils
import lycon
from utils.bundle_submissions import bundle_submissions_srgb_v1
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from torchmetrics.functional import structural_similarity_index_measure as ssim_func

parser = argparse.ArgumentParser(description='RGB denoising evaluation on DND dataset')
parser.add_argument('--input_dir', default='./datasets/Noise_var/test/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/denoising/dnd_rgb.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir+'matfile')
utils.mkdir(args.result_dir+'png')

test_dataset = get_test_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=8, drop_last=False)

model_restoration = DenoiseNet()

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()

def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def batch_SSIM(img1, img2, data_range=None):
    SSIM = []
    # print(img1.shape, img2.shape)
    # ssim_func = StructuralSimilarityIndexMeasure(data_range=255)
    for im1, im2 in zip(img1, img2):
        # unsqueeze(0) is for batch size
        im1 = im1.unsqueeze(0)
        im2 = im2.unsqueeze(0)
        
        ssim = ssim_func(im1, im2)
        SSIM.append(ssim)
        # print(ssim)
    return sum(SSIM)/len(SSIM)

def batch_PSNR(img1, img2, data_range=None):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

model_restoration=nn.DataParallel(model_restoration)

model_restoration.eval()
text_file = open(args.result_dir+"results.txt", "a")

with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_noisy = data_test[0].cuda()
        filenames = data_test[1]
        rgb_gt = data_test[2].cuda().permute(0, 3, 1, 2)
        gt_filenames = data_test[3]
        rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = torch.clamp(rgb_restored,0,1)
        rgb_test = torch.clamp(rgb_restored,0,1)
        
        # print(rgb_test.shape, rgb_gt.shape)
    
     
        rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        
        psnr = batch_PSNR(rgb_test, rgb_gt, 1.)
        ssim = batch_SSIM(rgb_test, rgb_gt, 1.)
        
        psnr_val_rgb.append(psnr)
        ssim_val_rgb.append(ssim)
        text_file.write("PSNR: %f SSIM: %f \n" % (psnr, ssim))

        if args.save_images:
            for batch in range(len(rgb_noisy)):
                denoised_img = img_as_ubyte(rgb_restored[batch])
                save_img(args.result_dir+'png/'+filenames[batch][:-4]+'.png', denoised_img)
                # save_file = os.path.join(args.result_dir+ 'matfile/', filenames[batch][:-4] +'.mat')
                # sio.savemat(save_file, {'Idenoised_crop': np.float32(rgb_restored[batch])})

psnr_val_rgb = sum(psnr_val_rgb)/len(test_dataset)
print("PSNR: ", psnr_val_rgb)
ssim_val_rgb = sum(ssim_val_rgb)/len(test_dataset)
print("SSIM: ", ssim_val_rgb)


