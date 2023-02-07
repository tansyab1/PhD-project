#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-07-10 14:38:39

'''
In this demo, we only test the model on one image of SIDD validation dataset.
The full validation dataset can be download from the following website:
    https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php
'''

import torch
from networks import UNetG, sample_generator
from scipy.io import loadmat
from skimage import img_as_float32, img_as_ubyte
from matplotlib import pyplot as plt
from utils import PadUNet


# build the network
net = UNetG(3, wf=32, depth=5).cuda()

# load the pretrained model
net.load_state_dict(torch.load('./model_states/DANet.pt', map_location='cpu')['G'])

# read the images
im_noisy_real = loadmat('./test_data/SIDD/noisy.mat')['im_noisy']
im_gt = loadmat('./test_data/SIDD/gt.mat')['im_gt']

# denoising
inputs = torch.from_numpy(img_as_float32(im_gt).transpose([2,0,1])).unsqueeze(0).cuda()
with torch.autograd.no_grad():
    padunet = PadUNet(inputs, dep_U=5)
    inputs_pad = padunet.pad()
    outputs_pad = sample_generator(net, inputs_pad)
    outputs = padunet.pad_inverse(outputs_pad)
    outputs.clamp_(0.0, 1.0)

im_noisy_fake = img_as_ubyte(outputs.cpu().numpy()[0,].transpose([1,2,0]))

plt.subplot(1,3,1)
plt.imshow(im_gt)
plt.title('Gt Image')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(im_noisy_real)
plt.title('Real Noisy Image')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(im_noisy_fake)
plt.title('Fake Noisy Image')
plt.axis('off')
plt.show()

