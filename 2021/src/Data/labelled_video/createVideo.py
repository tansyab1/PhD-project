
# import from addNoise.py 
import numpy as np
import cv2
# import csv
import os
import glob
from tqdm import tqdm
# from functools import reduce
from skimage.util import random_noise

# import all functions from addNoise.py
from addNoise import create_noise, addNoise
# import all functions from addUI.py
from addUI import addUI
# import all functions from addBlur.py
from addBlur import addBlur

# run the main function
if __name__ == "__main__":
    # addNoise()
    addUI()
    addBlur()