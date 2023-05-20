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

pathOut = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/main/KvasirCapsuleIQA/final/imgs/'

variance_list = []
entropy_list = []

# read all image in folder and calculate variance and entropy of each image
for image in os.listdir(pathOut):
    if image.endswith(".jpg"):
        # get image name
        image_name = image.split('.')[0]
        # load image
        img = cv2.imread(pathOut + image, cv2.IMREAD_GRAYSCALE)
        # print(np.max(img))
        # # convert image to grayscale without scale to 0-1
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # convert image to tensor
        # img = transforms.ToTensor()(img)

        # calculate variance and entropy of image
        variance = np.var(img)
        # print(variance)
        # calculate normalized histogram of image
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist / np.sum(hist)
        # calculate entropy of image
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        print(entropy)
        # add variance and entropy to list
        variance_list.append(variance)
        entropy_list.append(entropy)

# plot histogram of variance and entropy with mininumum border
# set border top and bottom of histogram to 0
# plt.ylim(0, 8)
# # set border left and right of histogram to 0
# plt.xlim(0, 5000)


plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.hist(variance_list, color='orange', edgecolor='black', bins=50)
# # save histogram to file .eps
# plt.savefig('/home/nguyentansy/DATA/PhD-work/PhD-project/2023/src/variance.png'
            # )
# x-axis label

# rotate x-axis label to 45 degree
# plt.xticks(rotation=45)
# # set x-axis range
# plt.xticks(np.arange(0, 5000, 500))
# plt.yticks(np.arange(0, 8, 1))
plt.xlabel('Entropy of images')
# frequency label
plt.ylabel('Frequency')

# save to tight layout
plt.tight_layout()
# save histogram to file .eps
# plt.savefig('/home/nguyentansy/DATA/PhD-work/PhD-project/2023/src/variance.eps',
#             format='eps')
plt.hist(entropy_list, color='red', edgecolor='black', bins=50)
# save histogram to file .eps
plt.savefig('/home/nguyentansy/DATA/PhD-work/PhD-project/2023/src/entropy.eps', format='eps')
