import cv2
import os
import matplotlib.pyplot as plt


def detectBlur(img_path, threhold):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = getVarianofLaplace(gray)
    if (var < threhold):
        return True
    else:
        return False


def readImagefromFolderpath(folder_path):
    img_list = []
    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            img_list.append(file)
    return img_list


def plotHistoram(arr):
    plt.hist(arr, bins=256)
    plt.show()


def getVarianofLaplace(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


if __name__ == '__main__':
    print(detectBlur("../../../../Downloads/blur.jpg", 0.01))
    var_list = []
    img_list = readImagefromFolderpath("../../../../Downloads/blur")
    for img in img_list:
        var_list.append(getVarianofLaplace(
            cv2.imread("../../../../Downloads/blur/" + img)))
    plotHistoram(var_list)
