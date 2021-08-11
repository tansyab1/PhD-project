
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import math
import os


def checkUnevenIllumination(image_path):
    """check the image whether have the uneven illumination
    Args:
        image_path ([type]): [description]
    """
    img = mpimg.imread(image_path)
    return img
