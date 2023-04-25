import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    return img


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])
##################################################################################################


def get_training_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir)


def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir)


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir):
        super(DataLoaderTrain, self).__init__()

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, 'groundtruth')))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))

        self.clean_filenames = [os.path.join(
            rgb_dir, 'groundtruth', x) for x in clean_files if is_image_file(x)]
        self.noisy_filenames = [os.path.join(
            rgb_dir, 'input', x) for x in noisy_files if is_image_file(x)]

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        clean = torch.from_numpy(np.float32(
            load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(
            load_img(self.noisy_filenames[tar_index])))

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        return clean, noisy, clean_filename, noisy_filename


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, 'groundtruth')))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))

        self.clean_filenames = [os.path.join(
            rgb_dir, 'groundtruth', x) for x in clean_files if is_image_file(x)]
        self.noisy_filenames = [os.path.join(
            rgb_dir, 'input', x) for x in noisy_files if is_image_file(x)]

        self.tar_size = len(self.clean_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        clean = torch.from_numpy(np.float32(
            load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(
            load_img(self.noisy_filenames[tar_index])))

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        return clean, noisy, clean_filename, noisy_filename
