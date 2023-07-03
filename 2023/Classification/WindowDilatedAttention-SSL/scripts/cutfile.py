import os


# select random 1000 images with same name from each folder and move them to a new folder
folder = ["input", "groundtruth"]
original_path = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/upload/"


def changedir(typedis):
    val_path = original_path + typedis + "/val/"
    train_path = original_path + typedis + "/train/"
    test_path = original_path + typedis + "/test/"

    cutfile(31, val_path, train_path)
    cutfile(47, val_path, test_path)


def cutfile(num=1000, original_path=original_path, destination_path=original_path):
    input_path = original_path + folder[0] + "/"
    groundtruth_path = original_path + folder[1] + "/"
    for i in range(num):
        filename = os.listdir(input_path)[i]
        os.rename(input_path + "/" + filename, destination_path + folder[0] + "/" + filename)
        os.rename(groundtruth_path + "/" + filename, destination_path + folder[1] + "/" + filename)


if __name__ == '__main__':
    # changedir("Noise")
    changedir("Blur")
    changedir("UI")
