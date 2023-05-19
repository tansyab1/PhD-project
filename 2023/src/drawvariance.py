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
        img = Image.open(pathOut + image)
        # convert image to grayscale
        img = img.convert('L')
        # convert image to tensor
        img = transforms.ToTensor()(img)
        print(np.mean(img.numpy()))
        # calculate variance and entropy of image
        variance = np.std(img.numpy())
        # entropy = -np.sum(img.numpy() * np.log2(img.numpy() + 1e-10))
        # add variance and entropy to list
        variance_list.append(variance)
        # entropy_list.append(entropy)

# plot histogram of variance and entropy with mininumum border
plt.hist(variance_list, color='blue', edgecolor='black')
# save histogram to file .eps
plt.savefig('/home/nguyentansy/DATA/PhD-work/PhD-project/2023/src/variance.png'
            )

# plt.hist(entropy_list, color='red', edgecolor='black')
# # save histogram to file .eps
# plt.savefig('/home/nguyentansy/DATA/PhD-work/PhD-project/2023/src/entropy.png'
#             )
