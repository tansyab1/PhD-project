#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-07-10 14:38:39

'''
In this demo, we only test the model on one image of SIDD validation dataset.
The full validation dataset can be download from the following website:
    https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php
'''

import argparse
import torch
from networks import UNetD
from scipy.io import loadmat
from skimage import img_as_float32, img_as_ubyte
from matplotlib import pyplot as plt
from utils import PadUNet
from glob import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='DANet+',
                    help="Model selection: DANet or DANet+, (default:DANet+)")
parser.add_argument('--save_dir', type=str, default='./results/',
                    help="The directory to save the denoised image, (default:./results/)")
parser.add_argument('--data_dir', type=str, default='./results/',
                    help="The directory to save the denoised image, (default:./results/)")
args = parser.parse_args()


# build the network
net = UNetD(3, wf=32, depth=5).cuda()

# load the pretrained model
if args.model.lower() == 'danet':
    net.load_state_dict(torch.load(
        './model_DANet/model_state_20.pt', map_location='cpu')['D'])
else:
    net.load_state_dict(torch.load(
        './model_states/DANetPlus.pt', map_location='cpu'))

# read the images
# im_noisy = loadmat('./test_data/SIDD/noisy.mat')['im_noisy']
# im_gt = loadmat('./test_data/SIDD/gt.mat')['im_gt']

path_all_noisy = glob(os.path.join(
    args.data_dir, 'input/*.jpg'), recursive=True)
path_all_noisy = sorted(path_all_noisy)
# print(os.path.join(args.data_dir, 'input/'))
path_all_gt = [x.replace('input', 'groundtruth') for x in path_all_noisy]
print('Number of big images: {:d}'.format(len(path_all_gt)))

for ii in range(len(path_all_gt)):
    # denoising
    inputs = torch.from_numpy(img_as_float32(plt.imread(
        path_all_noisy[ii])).transpose([2, 0, 1])[None, ]).cuda()
    with torch.autograd.no_grad():
        padunet = PadUNet(inputs, dep_U=5)
        inputs_pad = padunet.pad()
        outputs_pad = inputs_pad - net(inputs_pad)
        outputs = padunet.pad_inverse(outputs_pad)
        outputs.clamp_(0.0, 1.0)

    im_denoise = img_as_ubyte(outputs.cpu().numpy()[0, ].transpose([1, 2, 0]))
    # save the denoised image to the disk

    plt.imsave(os.path.join(args.save_dir, os.path.basename(path_all_noisy[ii])), im_denoise)

# plt.subplot(1,3,1)
# plt.imshow(im_noisy)
# plt.title('Noisy Image')
# plt.axis('off')
# plt.subplot(1,3,2)
# plt.imshow(im_gt)
# plt.title('Gt Image')
# plt.axis('off')
# plt.subplot(1,3,3)
# plt.imshow(im_denoise)
# plt.title('Denoised Image')
# plt.axis('off')
# plt.show()
