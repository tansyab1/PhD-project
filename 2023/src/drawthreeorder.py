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
from thirdentropy import thirdEntropy
from tqdm import tqdm

pathOut = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_videos/process/labelled_videos_process/main/KvasirCapsuleIQA/Reference'

# variance_list = []
entropy_list = []

# read all image in folder and calculate variance and entropy of each image
for image in tqdm(os.listdir(pathOut)):
    if image.endswith(".mp4"):
        # read video
        cap = cv2.VideoCapture(pathOut + '/' + image)
        # get total frame of video
        # check the first frame
        ret, frame1 = cap.read()
        
        # print("shape of frame1: ", frame1.shape )
        if ret == True:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            while cap.isOpened():
                # read grayscale image
                ret, frame2 = cap.read()
                
                # calculate entropy of image
                if ret == True:
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                    # print("processing frame: ", cap.get(cv2.CAP_PROP_POS_FRAMES))
                    entropy_list.append(thirdEntropy(frame1, frame2))
                    frame1 = frame2
                else:
                    break
        else:
            break
            

# plot histogram of variance and entropy with mininumum border
# set border top and bottom of histogram to 0
# plt.ylim(0, 8)
# # set border left and right of histogram to 0
# plt.xlim(0, 5000)


plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.hist(entropy_list, color='green', edgecolor='black', bins=50)
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
plt.xlabel('Third-Order Entropy')
# frequency label
plt.ylabel('Frequency')

# save to tight layout
plt.tight_layout()
# save histogram to file .eps
plt.savefig('/home/nguyentansy/DATA/PhD-work/PhD-project/2023/src/thirdentropy.eps',
            format='eps')
# plt.hist(entropy_list, color='red', edgecolor='black', bins=50)
# # save histogram to file .eps
# plt.savefig('/home/nguyentansy/DATA/PhD-work/PhD-project/2023/src/entropy.eps', format='eps')
# >>>>>>> 7aa644fa66640a0e21d0005c05bd151f49c92389
