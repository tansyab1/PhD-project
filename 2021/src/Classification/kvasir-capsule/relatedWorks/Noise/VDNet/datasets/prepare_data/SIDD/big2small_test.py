#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-18 10:26:59

import os
import numpy as np
import h5py as h5
from scipy.io import loadmat
import argparse
from glob import glob
import cv2

parser = argparse.ArgumentParser(prog='SIDD Test dataset Generation')
# The validation set of SIDD: ValidationNoisyBlocksSrgb.mat, ValidationGtBlocksSrgb.mat
parser.add_argument('--data_dir', default=None, type=str, metavar='PATH',
                                    help="path to save the validation set of SIDD, (default: None)")
args = parser.parse_args()

print('Validation: Saving the noisy blocks to hdf5 format!')
path_h5 = os.path.join(args.data_dir, 'small_imgs_test.hdf5')
if os.path.exists(path_h5):
    os.remove(path_h5)
# val_data_dict = loadmat(os.path.join(args.data_dir, 'ValidationNoisyBlocksSrgb.mat'))
# val_data_noisy = val_data_dict['ValidationNoisyBlocksSrgb']
# val_data_dict = loadmat(os.path.join(args.data_dir, 'ValidationGtBlocksSrgb.mat'))
# val_data_gt = val_data_dict['ValidationGtBlocksSrgb']
# num_img, num_block, _, _, _ = val_data_gt.shape

path_all_noisy = glob(os.path.join(args.data_dir, 'input/*.jpg'), recursive=True)
path_all_noisy = sorted(path_all_noisy)
# print(os.path.join(args.data_dir, 'input/'))
path_all_gt = [x.replace('input', 'groundtruth') for x in path_all_noisy]
print('Number of big images: {:d}'.format(len(path_all_gt)))


num_patch = 0
with h5.File(path_h5, 'w') as h5_file:
    for ii in range(len(path_all_gt)):
        if (ii+1) % 10 == 0:
            print('    The {:d} original images'.format(ii+1))
            
        im_noisy_int8 = cv2.imread(path_all_noisy[ii])[:, :, ::-1]
        
        im_noisy = cv2.imread(path_all_noisy[ii])[:, :, ::-1]
        im_gt = cv2.imread(path_all_gt[ii])[:, :, ::-1]
        imgs = np.concatenate((im_noisy, im_gt), axis=2)
        h5_file.create_dataset(name=str(num_patch), shape=imgs.shape, dtype=imgs.dtype, data=imgs)
        num_patch += 1
print('Finish!\n')


