

"""
## Learning Enriched Features for Real Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## ECCV 2020
## https://arxiv.org/abs/2003.06792
"""
from collections import OrderedDict

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import scipy.io as sio
from model import FCN
from psnr import batch_PSNR, batch_SSIM, save_img
from dataloader import get_validation_data
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(
    description='RGB deblurring evaluation on the validation set of Noisevar')
parser.add_argument('--input_dir', default='./dataset/UI_var/test/',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/',
                    type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/UIcorrection/models/FCN/model_best.pth',
                    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str,
                    help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=16, type=int,
                    help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true',
                    help='Save denoised images in result directory')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

os.makedirs(args.result_dir, exist_ok=True)

test_dataset = get_validation_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs,
                         shuffle=False, num_workers=8, drop_last=False)


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


model_restoration = FCN()

# load weights
load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()

model_restoration = nn.DataParallel(model_restoration)

model_restoration.eval()


with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test[0].cuda()
        rgb_noisy = data_test[1].cuda()
        # print(rgb_gt.shape)
        filenames = data_test[2]
        rgb_restored = model_restoration(rgb_noisy)
        # print(rgb_restored.shape)
        rgb_restored = torch.clamp(rgb_restored, 0, 1)

        psnr_val_rgb.append(batch_PSNR(rgb_restored, rgb_gt, 1.))
        ssim_val_rgb.append(batch_SSIM(rgb_restored, rgb_gt, 1.))

        rgb_gt = rgb_gt.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        if args.save_images:
            for batch in range(len(rgb_gt)):
                denoised_img = img_as_ubyte(rgb_restored[batch])
                save_img(args.result_dir +
                         filenames[batch][:-4] + '.png', denoised_img)

psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
ssim_val_rgb = sum(ssim_val_rgb)/len(ssim_val_rgb)
print("PSNR: %.2f " % (psnr_val_rgb))
print("SSIM: %.4f " % (ssim_val_rgb))
