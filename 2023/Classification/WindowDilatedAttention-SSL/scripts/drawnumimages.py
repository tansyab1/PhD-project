import os
import numpy as np
import matplotlib.pyplot as plt


# read the folder and show the number of image inside the name of the directory is the label
sum = 0


def drawnumimages(path):
    sum = 0
    names = []
    nums = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            names.append(name)
            num = len(os.listdir(os.path.join(root, name)))
            nums.append(num)

    # plot bar chart of the number of images with each class is one color
    # plt.figure(figsize=(10, 10))
    for i in range(len(names)):
        plt.bar(names[i], nums[i], color=np.random.rand(3, ))
        # put the number of images on the top of the bar
        plt.text(names[i], nums[i], nums[i], ha="center", va="bottom")
        sum += nums[i]
        
    plt.xticks(rotation=45)
    # set size of the text
    plt.tick_params(labelsize=10)
    # set size of the label
    plt.rcParams["axes.labelsize"] = 10
    # set size of the title
    plt.rcParams["axes.titlesize"] = 10
    # set the title and label
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.title("Number of images in each class")
    plt.show()
    # save the bar chart
    plt.savefig("numimages.eps", format="eps")


if __name__ == "__main__":
    path = "/Volumes/Macintosh HD - Data/These/vscode-workspace/Data/downstream"
    drawnumimages(path)
