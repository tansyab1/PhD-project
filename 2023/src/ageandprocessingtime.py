from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import numpy as np
import os
import cv2
# import PCA library
from sklearn.decomposition import PCA
# iport TSNE library
from sklearn.manifold import TSNE
# import matplotlib library
import matplotlib.pyplot as plt
from tqdm import tqdm


age = [25, 25, 60,	26,	25,	25,	24,	25,	24,	49,	35,	34,	50,	25,	46,	21,
       48,	54,	22,	49,	43,	36,	49,	45,	49,	39,	39,	50,	43,	51,	51,	33,	50,	49]

times = [2690,	1828,	2568,	1543,	3356,	3537,	3254,
         2272,	1715,	1669,	3832,	2678,	1753,	3734,
         3954,	2680,	2201,	3399,	3047,	3684,	3349,
         2353,	3488,	1918,	2954,	3126,	2181,	3758,
         2510,	2035,	3703,	1902,	2261,	3820]


# plot histogram of variance and entropy with mininumum border
# set font size of x-axis and y-axis to 20
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.hist(age, color='green', edgecolor='black', bins=50)
# save histogram to file .eps
plt.savefig(
    '/home/nguyentansy/DATA/PhD-work/PhD-project/2023/src/age.eps', format='eps')

# plot for times
plt.hist(times, color='brown', edgecolor='black', bins=50)
# save histogram to file .eps
plt.savefig(
    '/home/nguyentansy/DATA/PhD-work/PhD-project/2023/src/times.eps', format='eps')
