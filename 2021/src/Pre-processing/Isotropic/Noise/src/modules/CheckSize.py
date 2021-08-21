# import numpy as np
import cv2
# import math
import matplotlib.pyplot as plt
# import os
# import pandas as pd
import glob
import seaborn as sns
from tqdm import tqdm
import matplotlib as mpl
# import warnings; warnings.filterwarnings(action='once')

large = 22
med = 16
small = 12
params = {'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
# %matplotlib inline

# Version
print(mpl.__version__)  # > 3.0.0
print(sns.__version__)  # > 0.9.0


def readImagefromFolder(folder="/home/nguyentansy/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/data/labeled-images/"):
    heights = []
    widths = []
    for filename in tqdm(glob.glob("%s/*/pathological-findings/*/*" % folder)):
        img = cv2.imread(filename, 0)
        height = img.shape[0]
        width = img.shape[1]
        heights.append(height)
        widths.append(width)
    return heights, widths


def plotHistogram(heights, widths):
    """
    Plot the histogram of the array.

    Parameters
    ----------
    arr : ndarray
        Array to plot the histogram.

    """

    # Create Fig and gridspec
    fig = plt.figure(figsize=(16, 10), dpi=300)
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

    # Define the axes
    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

    # Scatterplot on main ax
    ax_main.scatter(widths, heights, s=40, c='r', marker=".", label="Width x Height",
                    alpha=0.5, edgecolors='gray', linewidths=.5)

    # histogram on the right
    ax_bottom.hist(widths, 40, histtype='stepfilled',
                   orientation='vertical', color='blue')
    ax_bottom.invert_yaxis()

    # histogram in the bottom
    ax_right.hist(heights, 40, histtype='stepfilled',
                  orientation='horizontal', color='green')

    # Decorations
    ax_main.set(title='Scatterplot with Histograms \n width vs height',
                xlabel='width', ylabel='height')
    ax_main.title.set_fontsize(10)
    for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
        item.set_fontsize(10)

    # xstick = np.arange(0, 2000, 100)
    # ystick = np.arange(0, 1200, 100)
    # ax_main.set_xticks(xstick)
    # ax_main.set_yticks(ystick)
    # xlabels = ax_main.get_xticks().tolist()
    # ylabels = ax_main.get_yticks().tolist()
    # ax_main.set_xticklabels(xlabels, rotation=45)
    # ax_main.set_yticklabels(ylabels)
    # plt.show()

    # # Draw Plot
    # plt.figure(figsize=(13, 10), dpi=80)
    # sns.histplot(arr, color="g",
    #              label="noise standard deviation")
    # plt.ylim(0, 0.35)
    # plt.xticks(np.arange(0, 1.5, 0.05), rotation=45)
    # Decoration
    plt.title('size analysis', fontsize=12)
    # plt.legend()
    filesave = "/home/nguyentansy/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/Noise/src/denoising_rgb/results/sizehist.png"
    plt.savefig(filesave)


if __name__ == "__main__":
    height, width = readImagefromFolder()
    plotHistogram(heights=height, widths=width)
