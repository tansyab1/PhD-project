from torchvision import datasets
import pickle
import os
import random
# from dataset.Dataloader_with_path import ImageFolderWithPaths


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # read pickle file
    def read_pickle(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)

    # get the reference image using datasets.ImageFolder class
    def get_reference_image(self, path, ref_paths):
        # get the reference image path
        # find the file path of the ref_paths with the same basename as the path
        ref_path = [x for x in self.ref_images_loader.imgs if os.path.basename(
            x[0]) == os.path.basename(path)]
        # get the reference image index
        ref_path_index = self.ref_images_loader.imgs.index(ref_path[0])
        return ref_path_index

    # create triplet of anchor, positive and negative images

    def create_triplet(self, index):

        # read pickle file
        pickle_file = self.read_pickle(self.pickle_file)

        # the image file path
        path = self.imgs[index][0]
        # get the noise level from the dictionary
        anchor_label = self.read_pickle(self.pickle_file)[
            os.path.basename(path)]

        # get the reference image path
        ref_index = self.get_reference_image(path, self.ref_paths)

        # get the list of the positive images with the same label as the anchor image
        # but different from the anchor image
        positive_images = [x for x in self.imgs if pickle_file[os.path.basename(
            x[0])] == anchor_label and x[0] != path]
        # get the list of the negative images with the different label than the anchor image
        negative_images = [
            x for x in self.imgs if pickle_file[os.path.basename(x[0])] != anchor_label]

        # get a random positive image
        positive_image = random.choice(positive_images)
        # get a random negative image
        negative_image = random.choice(negative_images)

        # get index of the positive image in the dataset
        positive_index = self.imgs.index(positive_image)
        # get index of the negative image in the dataset
        negative_index = self.imgs.index(negative_image)

        # return the triplet including the anchor, positive and negative images
        return positive_index, negative_index, ref_index

    # init the class
    def __init__(self, root, ref_paths, pickle_file, transform=None):
        super(ImageFolderWithPaths, self).__init__(root, transform)
        self.pickle_file = pickle_file
        self.ref_paths = ref_paths
        # object of the datasets.ImageFolder class to get the reference image
        self.ref_images_loader = datasets.ImageFolder(ref_paths, transform)

    # override the __getitem__ method. this is the method that dataloader calls

    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # print(original_tuple)

        # create triplet of anchor, positive and negative images
        positive_index, negative_index, ref_index = self.create_triplet(index)
        positive = super(ImageFolderWithPaths,
                         self).__getitem__(positive_index)
        negative = super(ImageFolderWithPaths,
                         self).__getitem__(negative_index)

        # get the reference image
        ref = self.ref_images_loader.__getitem__(ref_index)

        # make a new tuple that includes original and the path
        tuple_with_path = (
            original_tuple + (positive[0],) + (negative[0],) + (ref[0],))

        return tuple_with_path


if __name__ == "__main__":

    root_path = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/Noise/5/0"
    pickle_file = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/distorted_images/Noise_var/noise_dict.pkl"
    ref_paths = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/Noise/5/0"

    test = ImageFolderWithPaths(
        root_path, ref_paths, pickle_file, transform=None)
    print(test[0][3])
