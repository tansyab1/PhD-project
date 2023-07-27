import torch
import torchvision
from augmentations import augmentation
import torchvision.transforms as transforms


class initialize_dataset:
    def __init__(self, image_resolution=224, batch_size=128, path='./data'):
        self.image_resolution = image_resolution
        self.batch_size = batch_size
        self.path = path

    def load_dataset(self, transform=False):

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(
            (self.image_resolution, self.image_resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        # inside path having 3 folders: train, test, val. load train and test

        train_dataset = torchvision.datasets.ImageFolder(root=self.path + '/train',
                                                         transform=transform)
        
        test_dataset = torchvision.datasets.ImageFolder(root=self.path + '/test',
                                                        transform=transform)

        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True)

        return train_dataloader, test_dataloader
