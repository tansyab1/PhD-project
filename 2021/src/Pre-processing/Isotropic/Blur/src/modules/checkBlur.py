import cv2
import os
import matplotlib.pyplot as plt


def detectBlur(img_path, threhold):
    """detect blur of the image

    Args:
        img_path (string): image path
        threhold (float): threshold of blur

    Returns:
        Bool: yes or no
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = getVarianofLaplace(gray)
    if (var < threhold):
        return True
    else:
        return False


def readImagefromFolderpath(folder_path):
    """read all images from folder
    Args:
        folder_path (string): folder
    Returns:
        list: list of image path
    """
    img_list = []
    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            img_list.append(file)
    return img_list


def plotHistoram(arr):
    """plot histogram of the array
    Args:
        arr (list): array
    Returns:
        None
    """
    plt.hist(arr, bins=256)
    plt.show()


def getVarianofLaplace(img):
    """get variance of laplace of the image
    Args:
        img (numpy array): image
    Returns:
        float: variance of laplace
    """
    return cv2.Laplacian(img, cv2.CV_64F).var()


if __name__ == '__main__':
    print(detectBlur("../../../../Downloads/blur.jpg", 0.01))
    var_list = []
    img_list = readImagefromFolderpath("../../../../Downloads/blur")
    for img in img_list:
        var_list.append(getVarianofLaplace(
            cv2.imread("../../../../Downloads/blur/" + img)))
    plotHistoram(var_list)
