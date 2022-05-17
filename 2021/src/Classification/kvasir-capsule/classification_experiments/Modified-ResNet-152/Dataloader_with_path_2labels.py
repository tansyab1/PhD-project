import torch
from torchvision import datasets
import pickle
import os
# from dataset.Dataloader_with_path import ImageFolderWithPaths






class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # read pickle file
    def read_pickle(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    
    # init the class
    def __init__(self, root, pickle_file):
        super(ImageFolderWithPaths, self).__init__(root)
        self.pickle_file = pickle_file

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # get the noise level from the dictionary
        noise_level = self.read_pickle(self.pickle_file)[os.path.basename(path)]
        
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (noise_level,))
        return tuple_with_path


if __name__=="__main__":

    root_path = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/Noise/5/0"
    pickle_file = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/distorted_images/Noise_var/noise_dict.pkl"

    test = ImageFolderWithPaths(root_path, pickle_file)
    print(test[0][2])