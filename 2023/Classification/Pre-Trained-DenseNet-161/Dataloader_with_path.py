# import torch
from torchvision import datasets
# from dataset.Dataloader_with_path import ImageFolderWithPaths


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


if __name__ == "__main__":

    root_path = "./dataset/Pseudo_folds/0/"

    test = ImageFolderWithPaths(root_path)

    # show the name of each group in the dataset
    # print(test.classes)
    # show the label corresponding to each class
    print(test.class_to_idx)
    # data length of the inference dataset is 2785829
