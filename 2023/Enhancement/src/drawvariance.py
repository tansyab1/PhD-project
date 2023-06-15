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
import seaborn as sns
from secondEntropy import secondEntropy
from tqdm import tqdm

pathOut = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/main/KvasirCapsuleIQA/Reference/imgs/'

variance_list = []
entropy_list = []

# read all image in folder and calculate variance and entropy of each image
for image in tqdm(os.listdir(pathOut)):
    if image.endswith(".jpg"):
        # get image name
        image_name = image.split('.')[0]
        # load image
        img = cv2.imread(pathOut + image, cv2.IMREAD_GRAYSCALE)
        # calculate variance and entropy of image
        entropy_list.append(secondEntropy(img))
        # add variance and entropy to list
        # variance_list.append(variance)
        # entropy_list.append(entropy)

# plot histogram of variance and entropy with mininumum border
# set border top and bottom of histogram to 0
# plt.ylim(0, 8)
# # set border left and right of histogram to 0
# plt.xlim(0, 5000)


plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.hist(entropy_list, color='orange', edgecolor='black', bins=50)
# # save histogram to file .eps
# plt.savefig('/home/nguyentansy/DATA/PhD-work/PhD-project/2023/src/variance.png'
            # )
# x-axis label

# =======
# rotate x-axis label to 45 degree
# plt.xticks(rotation=45)
# # set x-axis range
# plt.xticks(np.arange(0, 5000, 500))
# plt.yticks(np.arange(0, 8, 1))
plt.xlabel('Second-Order Entropy')
# frequency label
plt.ylabel('Frequency')

# save to tight layout
plt.tight_layout()
# save histogram to file .eps
plt.savefig('/home/nguyentansy/DATA/PhD-work/PhD-project/2023/src/secondentropy.eps',
            format='eps')
# plt.hist(entropy_list, color='red', edgecolor='black', bins=50)
# # save histogram to file .eps
# plt.savefig('/home/nguyentansy/DATA/PhD-work/PhD-project/2023/src/entropy.eps', format='eps')
# >>>>>>> 7aa644fa66640a0e21d0005c05bd151f49c92389
