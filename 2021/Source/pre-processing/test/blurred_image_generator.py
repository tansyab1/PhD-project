# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 08:27:25 2017

@author: user
"""

import os
from PIL import Image
from pyblur import GaussianBlur_random, DefocusBlur_random
from pyblur import BoxBlur_random, LinearMotionBlur_random
from pyblur import PsfBlur_random
from tqdm import tqdm
from itertools import product
import argparse

def append_tag(name, tag):
    tokens = name.rsplit('.', 1)
    #tokens[0]: xxx, tokens[1]: jpg/jpeg
    return '.'.join([tokens[0]+'_'+tag, tokens[1]])

def save_img(img, fname):
    with open(fname, 'wb') as f:
        img.save(f)

parser = argparse.ArgumentParser()
parser.add_argument("--imagedir", type=str, 
                    help="the folder of your images",
                    default="images")
parser.add_argument("--blurreddir", type=str, 
                    help="the folder of generated blurred images",
                    default="blurred")
parser.add_argument("--blurred_num", type=int,
                    help="the number of blurred images to be generated " + 
                    "per type per image",
                    default=5)

args = parser.parse_args()
imagedir = args.imagedir
blurreddir = args.blurreddir
blurred_num = args.blurred_num

if not os.path.isdir(blurreddir):
    os.makedirs(blurreddir)
    
for fimg, count in tqdm(product(os.listdir(imagedir), range(blurred_num))):
    img = Image.open("{}/{}".format(imagedir, fimg))
    
    blurred = GaussianBlur_random(img)
    save_img(blurred, "{}/{}".format(blurreddir, 
             append_tag(fimg, "GaussianBlur_{}".format(count))))
    
    blurred = DefocusBlur_random(img)
    save_img(blurred, "{}/{}".format(blurreddir, 
             append_tag(fimg, "DefocusBlur_{}".format(count))))
    
    blurred = BoxBlur_random(img)
    save_img(blurred, "{}/{}".format(blurreddir, 
             append_tag(fimg, "BoxBlur_{}".format(count))))

    blurred = LinearMotionBlur_random(img)
    save_img(blurred, "{}/{}".format(blurreddir, 
             append_tag(fimg, "LinearMotionBlur_{}".format(count))))
    
    blurred = PsfBlur_random(img)
    save_img(blurred, "{}/{}".format(blurreddir, 
             append_tag(fimg, "PsfBlur_{}".format(count))))
    
    img.close()