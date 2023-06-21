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


# BRISQUE = [4426, 192, 36, 1, 50, 1111, 201, 98, 9918]
# # DE = [12, 19, 1134, 150, 7634, 398, 47, 797, 5609]
# NIQE = [41, 81, 821, 77, 1093, 4, 1047, 19, 12817]


# method = ['BM3D', 'CycleISP', 'DANet', 'DIPNet',
#           'VDNet', 'MPRNet', 'MIRNet', 'Uformer', 'TCFA']

BRISQUE2 = [6, 41, 1134, 26, 3217, 1502, 2157, 7917]
DE2 = [31, 525, 6, 505, 25, 14421, 64, 426]
NIQE2 = [15, 26, 73, 19, 23, 2456, 5160, 8228]

method2 = ['TV', 'DBGAN', 'DeblurGANv2', 'DMPHN',
           'MPRNet', 'MIRNet', 'Uformer', 'TCFA']


# plot group bar chart where x1 is BRISQUE, x2 is DE, x3 is NIQE of each method
# set width of bar
barWidth = 0.2

# # plot bars group1
# plt.bar([1, 2, 3, 4, 5, 6, 7, 8, 9], BRISQUE, color='grey',
#         width=barWidth,  label='BRISQUE')
# # plot bars group2
# plt.bar([1 + barWidth, 2 + barWidth, 3 + barWidth, 4 + barWidth, 5 + barWidth, 6 + barWidth, 7 + barWidth, 8 + barWidth, 9 + barWidth], NIQE, color='orange',
#         width=barWidth,  label='NIQE')
# # # plot bars group3
# # plt.bar([1 + barWidth * 2, 2 + barWidth * 2, 3 + barWidth * 2, 4 + barWidth * 2, 5 + barWidth * 2, 6 + barWidth * 2, 7 + barWidth * 2, 8 + barWidth * 2, 9 + barWidth * 2], NIQE, color='green',
# #         width=barWidth,  label='NIQE')
# # # draw vertical line after each bar in group3
# for i in range(1, 10):
#     plt.axvline(x=i + barWidth * 2+ 0.125, color='black', linestyle='--', linewidth=0.5)

# # set the position of the x ticks
# plt.xticks([r + barWidth*0.5 for r in range(1, 10)], method)
# # rotate labels 45 degree
# plt.xticks(rotation=45)
# # set y label
# plt.ylabel('number of images')

# # set title of plot
# plt.title('Denoising')
# # set x label
# plt.xlabel('Method')


# # show legend on the top right of plot
# plt.legend(loc='upper left')
# # tight layout
# plt.tight_layout()
# # save plot to file .eps
# plt.savefig('2023/Enhancement/src/denoising-node.eps', format='eps')

# # deblurring for method2

# # plot group bar chart where x1 is BRISQUE, x2 is DE, x3 is NIQE of each method
# # set width of bar
# barWidth = 0.2

# # plot bars group1
# plt.bar([1, 2, 3, 4, 5, 6, 7, 8], BRISQUE2, color='grey',
#         width=barWidth,  label='BRISQUE')
# # plot bars group2
# plt.bar([1 + barWidth, 2 + barWidth, 3 + barWidth, 4 + barWidth, 5 + barWidth, 6 + barWidth, 7 + barWidth, 8 + barWidth], NIQE2, color='orange',
#         width=barWidth,  label='NIQE')
# # plot bars group3
# # plt.bar([1 + barWidth * 2, 2 + barWidth * 2, 3 + barWidth * 2, 4 + barWidth * 2, 5 + barWidth * 2, 6 + barWidth * 2, 7 + barWidth * 2, 8 + barWidth * 2], NIQE2, color='green',
# #         width=barWidth,  label='NIQE')

# # draw vertical line after each bar in group3
# for i in range(1, 9):
#     plt.axvline(x=i + barWidth * 2+ 0.125, color='black', linestyle='--', linewidth=0.5)

# # set the position of the x ticks
# plt.xticks([r + barWidth*0.5 for r in range(1, 9)], method2)
# # rotate labels 45 degree
# plt.xticks(rotation=45)
# # set y label
# plt.ylabel('number of images')

# # set title of plot
# plt.title('Deblurring')
# # set x label
# plt.xlabel('Method')


# # show legend on the top right of plot
# plt.legend(loc='upper left')
# # tight layout
# plt.tight_layout()
# # save plot to file .eps
# plt.savefig('2023/Enhancement/src/deblurring-node.eps', format='eps')


BRISQUE3 = [145, 65, 78, 215, 3005, 3023, 2248, 3232, 3939]
# DE3 = [2210, 956, 5350, 307, 99, 125, 139, 1981, 4833]
NIQE3 = [510, 36, 20, 158, 18, 223, 2291, 1976, 10768]
LOE3 = [645, 15, 967, 310, 8852, 224, 1168, 784, 3035]

method3 = ['AFGT', 'FLM', 'LIME', 'RetinexNet',
           'EnlightenGAN', 'MIRNet', 'FCN', 'Uformer', 'TCFA']

# plot group bar chart where x1 is BRISQUE, x2 is DE, x3 is NIQE, x4 is LOE of each method
# set width of bar
barWidth = 0.2

# plot bars group1
plt.bar([1, 2, 3, 4, 5, 6, 7, 8, 9], BRISQUE3, color='grey',
        width=barWidth,  label='BRISQUE')
# plot bars group2
plt.bar([1 + barWidth, 2 + barWidth, 3 + barWidth, 4 + barWidth, 5 + barWidth, 6 + barWidth, 7 + barWidth, 8 + barWidth, 9 + barWidth], LOE3, color='orange',
        width=barWidth,  label='LOE')
# plot bars group3
plt.bar([1 + barWidth * 2, 2 + barWidth * 2, 3 + barWidth * 2, 4 + barWidth * 2, 5 + barWidth * 2, 6 + barWidth * 2, 7 + barWidth * 2, 8 + barWidth * 2, 9 + barWidth * 2], NIQE3, color='green',
        width=barWidth,  label='NIQE')
# # plot bars group4
# plt.bar([1 + barWidth * 3, 2 + barWidth * 3, 3 + barWidth * 3, 4 + barWidth * 3, 5 + barWidth * 3, 6 + barWidth * 3, 7 + barWidth * 3, 8 + barWidth * 3, 9 + barWidth * 3], LOE3, color='blue',
#         width=barWidth,  label='LOE')

# draw vertical line after each bar in group4 
for i in range(1, 10):
        plt.axvline(x=i + barWidth * 3 + 0.125, color='black', linestyle='--', linewidth=0.5)

                


# set the position of the x ticks
plt.xticks([r + 1*barWidth for r in range(1, 10)], method3)
# rotate labels 45 degree
plt.xticks(rotation=45)
# set y label
plt.ylabel('number of images')

# set title of plot
plt.title('Uneven Illumination Correction')
# set x label
plt.xlabel('Method')


# show legend on the top right of plot
plt.legend(loc='upper left')
# tight layout
plt.tight_layout()
# save plot to file .eps
plt.savefig('2023/Enhancement/src/uneven-illumination-node.eps', format='eps')  