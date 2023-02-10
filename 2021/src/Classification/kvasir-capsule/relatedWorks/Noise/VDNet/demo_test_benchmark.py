#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-05-16 16:20:01

import sys

import numpy as np
import torch
from networks import VDN
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage import img_as_float32, img_as_ubyte
from utils import load_state_dict_cpu
from matplotlib import pyplot as plt
import time
from scipy.io import loadmat
import argparse
from glob import glob
import os
sys.path.append('./')

parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, default='DANet+',
#                     help="Model selection: DANet or DANet+, (default:DANet+)")
parser.add_argument('--save_dir', type=str, default='./results/',
                    help="Thoised image, (default:./results/)")
parser.add_argument('--data_dir', type=str, default='./results/',
                    help="The directory to save the denoised image, (default:./results/)")
args = parser.parse_args()

use_gpu = True
C = 3
dep_U = 4


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(255 / mse)


# load the pretrained model
print('Loading the Model')
checkpoint = torch.load('./model/model_state_20')
net = VDN(C, dep_U=dep_U, wf=64)
if use_gpu:
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(checkpoint)
else:
    load_state_dict_cpu(net, checkpoint)
net.eval()

f = open('results.txt', 'a')

path_all_noisy = glob(os.path.join(
    args.data_dir, 'input/*.jpg'), recursive=True)
path_all_noisy = sorted(path_all_noisy)
# print(os.path.join(args.data_dir, 'input/'))
path_all_gt = [x.replace('input', 'groundtruth') for x in path_all_noisy]
print('Number of big images: {:d}'.format(len(path_all_gt)))

psnr_all = 0
ssim_all = 0

for ii in range(len(path_all_gt)):
    # denoising
    img = plt.imread(path_all_noisy[ii])
    name = path_all_noisy[ii].split('/')[-1]
    H, W, _ = img.shape
    if H % 2**dep_U != 0:
        H -= H % 2**dep_U
    if W % 2**dep_U != 0:
        W -= W % 2**dep_U
    img = img[:H, :W, ]
    inputs = torch.from_numpy(img_as_float32(
        img).transpose([2, 0, 1])[np.newaxis, ])
    # if use_gpu:
    #     inputs = inputs.cuda()
    #     print('Begin Testing on GPU')
    # else:
    #     print('Begin Testing on CPU')
    with torch.autograd.set_grad_enabled(False):
        torch.cuda.synchronize()
        tic = time.perf_counter()
        phi_Z = net(inputs, 'test')
        torch.cuda.synchronize()
        toc = time.perf_counter()
        err = phi_Z.cpu().numpy()
    if use_gpu:
        inputs = inputs.cpu().numpy()
    else:
        inputs = inputs.numpy()
    # print('Finish, time: {:.2f}'.format(toc-tic))
    im_denoise = inputs - err[:, :C, ]
    im_denoise = np.transpose(im_denoise.squeeze(), (1, 2, 0))
    im_denoise = img_as_ubyte(im_denoise.clip(0, 1))
    inputs = np.transpose(inputs.squeeze(), (1, 2, 0))
    inputs = img_as_ubyte(inputs.clip(0, 1))
    psnr_im = psnr(im_denoise, inputs)
    psnr_all += psnr_im
    ssim_im = ssim(im_denoise, plt.imread(path_all_gt[ii]), multichannel=True)
    ssim_all += ssim_im
    f.write('{}, PSNR: {}, SSIM: {}\n '.format(name, psnr_im, ssim_im))
    print('{}, PSNR: {}, SSIM: {}'.format(name, psnr_im, ssim_im))
    # save the denoised image
    plt.imsave(os.path.join(args.save_dir, name), im_denoise)

print('Finish')
print('Average PSNR: {}, SSIM: {}'.format(psnr_all/len(path_all_gt), ssim_all/len(path_all_gt)))

# move the denoised image to the save_dir
os.system('mv {} {}'.format("results.txt", args.save_dir))


# plt.subplot(121)
# plt.imshow(im_noisy)
# plt.title('Noisy Image')
# plt.subplot(122)
# plt.imshow(im_denoise)
# plt.title('Denoised Image')
# plt.show()
