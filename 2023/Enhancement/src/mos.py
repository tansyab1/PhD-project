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


MOSexpertnoise = [1.267, 1.9755,	2.821,	3.535]
MOSexpertdfblur = [1.533, 2.221,  3.134, 3.741]
MOSexpertmotionblur = [1.2495, 2.2915, 2.6455, 3.642]
MOSexpertui = [1.5205, 2.084, 3.2495, 3.8465]

MOSnonexpertnoise = [1.172, 1.8555, 2.781, 3.53]
MOSnonexpertdfblur = [1.478, 2.141, 3.164, 3.721]
MOSnonexpertmotionblur = [1.1395, 2.2015, 2.6905, 3.582]
MOSnonexpertui = [1.3655, 2.049, 3.2145, 3.8515]


# plot group bar chart
# set width of bar
barWidth = 0.2
# set height of bar
bars1 = [1.172, 1.8555, 2.781, 3.53]
bars2 = [1.478, 2.141, 3.164, 3.721]
bars3 = [1.1395, 2.2015, 2.6905, 3.582]
bars4 = [1.3655, 2.049, 3.2145, 3.8515]
colors = ['grey', 'orange', 'green', 'purple']
markerscolors = ['black', 'red', 'olive', 'blue']
# set color of bars
# plt.bar([1, 2, 3, 4], bars1, color=colors[0], edgecolor='black', width=barWidth)
# plt.bar([1 + barWidth, 2 + barWidth, 3 + barWidth, 4 + barWidth], bars2,
#         color=colors[1], edgecolor='black', width=barWidth)
# plt.bar([1 + barWidth*2, 2 + barWidth*2, 3 + barWidth*2, 4 + barWidth*2],
#         bars3, color=colors[2], edgecolor='black', width=barWidth)

# change transparency of bars (alpha value)


# plot bars group1
plt.bar([1, 2, 3, 4], bars1, color=colors[0],
        width=barWidth,  label='Non-Expert Noise')
# plot bars group2
plt.bar([1 + barWidth, 2 + barWidth, 3 + barWidth, 4 + barWidth], bars2, color=colors[1],
        width=barWidth,  label='Non-Expert Defocus blur')
# plot bars group3
plt.bar([1 + barWidth*2, 2 + barWidth*2, 3 + barWidth*2, 4 + barWidth*2],
        bars3, color=colors[2], width=barWidth,  label='Non-Expert Motion blur')
# plot bars group4
plt.bar([1 + barWidth*3, 2 + barWidth*3, 3 + barWidth*3, 4 + barWidth*3], bars4, color=colors[3],
        width=barWidth,  label='Non-Expert Uneven illumination')

# plot the point of MOS of each group on the top of bar
plt.plot([1, 2, 3, 4], MOSexpertnoise, 'o', color=markerscolors[0], label='Expert Noise')
plt.plot([1 + barWidth, 2 + barWidth, 3 + barWidth, 4 + barWidth],
         MOSexpertdfblur, 'o', color=markerscolors[1], label='Expert Defocus blur')
plt.plot([1 + barWidth*2, 2 + barWidth*2, 3 + barWidth*2, 4 + barWidth*2],
         MOSexpertmotionblur, 'o', color=markerscolors[2], label='Expert Motion blur')
plt.plot([1 + barWidth*3, 2 + barWidth*3, 3 + barWidth*3, 4 + barWidth*3],
         MOSexpertui, 'o', color=markerscolors[3], label='Expert Uneven illumination')

# change the color map of bar


# set the size of x and y label
# plt.rc('xtick', labelsize=15)
# plt.rc('ytick', labelsize=15)
# each group has 4 bars with the same barwidth
# put the x ticks in the middle of the group bars
plt.xticks([1 + 1.5*barWidth, 2 + 1.5*barWidth, 3 + 1.5*barWidth, 4 + barWidth*1.5], [
    'Level 1', 'Level 2', 'Level 3', 'Level 4'])

plt.ylabel('MOS', fontsize=15)


# show legend on the top right of plot
plt.legend(loc='upper left')
# tight layout
plt.tight_layout()
# save plot to file .eps
plt.savefig('2023/src/MOS.eps', format='eps')
