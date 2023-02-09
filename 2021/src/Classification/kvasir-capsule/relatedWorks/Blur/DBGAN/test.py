import os
import sys
import logging
import time
import argparse
import numpy as np
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
opt = option.dict_to_nonedict(opt)

util.setup_logger(None, opt['path']['log'], 'test.log', level=logging.INFO, screen=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))
# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# Create model
model = create_model(opt)

# open text file to write PSNR and SSIM
f = open("psnr_ssim.txt", "a")

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)
    

    for data in test_loader:
        
        model.feed_data(data, False)
        img_path = data['LR_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        model.test()  # test
        visuals = model.get_current_visuals(False)

        sr_img = util.tensor2img(visuals['SR'])  # uint8
        ground_truth = data['HR'][0].numpy().transpose(1, 2, 0)
        
        # calculate PSNR and SSIM
        # print('calculating PSNR and SSIM...')
        psnr = util.calculate_psnr(sr_img, ground_truth)
        ssim = util.calculate_ssim(sr_img, ground_truth)
        
        f.write(img_name + "PSNR: " + str(psnr) + "SSIM: " + str(ssim))

        # save images
        suffix = opt['suffix']
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + '.png')
        else:
            save_img_path = os.path.join(dataset_dir, img_name + '.png')
        print('Saving image [{:s}] ...'.format(save_img_path))
        util.save_img(sr_img, save_img_path)

# move the text file to the results folder
os.rename("psnr_ssim.txt", os.path.join(dataset_dir, "psnr_ssim.txt"))

       
       
       