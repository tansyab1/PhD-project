from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
# from __future__ import print_function, division

import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
# import TripletLoss as TripletLoss
from sklearn.manifold import TSNE
# from tsnecuda import TSNE
import matplotlib.pyplot as plt
import gc
import os
import copy
import pandas as pd
import numpy as np
import itertools
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum
import psutil
import GPUtil
from scipy.io import savemat
from torch.utils.tensorboard import SummaryWriter

from torchmetrics import StructuralSimilarityIndexMeasure

from tqdm import tqdm
from torchsummary import summary
from torch.autograd import Variable

from utils.Dataloader_with_path_2labels_ref import ImageFolderWithPaths as dataset

import string
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

# ======================================
# Get and set all input parameters
# ======================================
parser = argparse.ArgumentParser()

# Hardware
parser.add_argument("--device", default="gpu", help="Device to run the code")
parser.add_argument("--device_id", type=int, default=0, help="")


# store current python file
parser.add_argument("--py_file", default=os.path.abspath(__file__))


# Directories
parser.add_argument("--data_root",
                    default="ExperimentalDATA/Noise_var/",
                    help="data root directory")

parser.add_argument("--ref_root",
                    default="ExperimentalDATA/ref/",
                    help="data root directory")

parser.add_argument("--pkl_root",
                    default="src/dict/noise_dict.pkl",
                    help="pkl root directory")


parser.add_argument("--data_to_inference",
                    default="ExperimentalDATA/interference/",
                    help="Data folder with one subfolder which containes images to do inference")

parser.add_argument("--out_dir",
                    default="output/Dif-level/diffusion/",
                    help="Main output dierectory")

parser.add_argument("--mat_dir",
                    default="output/Dif-level/diffusion/",
                    help="Main output dierectory")

parser.add_argument("--tensorboard_dir",
                    default="output/tensorboard/Dif-level/diffusion/",
                    help="Folder to save output of tensorboard")

# Hyper parameters
parser.add_argument("--bs", type=int, default=32, help="Mini batch size")

parser.add_argument("--lr", type=float, default=0.01,
                    help="Learning rate for training")

parser.add_argument("--num_workers", type=int, default=10,
                    help="Number of workers in dataloader")

parser.add_argument("--weight_decay", type=float,
                    default=1e-5, help="weight decay of the optimizer")

parser.add_argument("--momentum", type=float, default=0.9,
                    help="Momentum of SGD function")

parser.add_argument("--lr_sch_factor", type=float, default=0.1,
                    help="Factor to reduce lr in the scheduler")

parser.add_argument("--lr_sch_patience", type=int, default=10,
                    help="Num of epochs to be patience for updating lr")

# Action handling
parser.add_argument("--num_epochs",
                    type=int,
                    default=50,
                    help="Numbe of epochs to train")
# parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch in retraining")
parser.add_argument("--action",
                    type=str,
                    help="Select an action to run",
                    choices=["train", "retrain", "test", "check", "prepare", "inference"])

parser.add_argument("--checkpoint_interval",
                    type=int,
                    default=25,
                    help="Interval to save checkpoint models")

parser.add_argument("--val_fold",
                    type=str,
                    default="1",
                    help="Select the validation fold", choices=["0", "1"])

parser.add_argument("--all_folds",
                    default=["0", "1"],
                    help="list of all folds available in data folder")

opt = parser.parse_args()

def load_data(data_dir="./data"):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    return trainset, testset

class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, l1)
        # self.fc2 = nn.Linear(l1, l2)
        # self.fc3 = nn.Linear(l2, 10)
        
        self.encoder_mlp = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, x, positive, negative):
        # x = x.cuda(1)
        # self.encoder = self.encoder.cuda(1)
        # encoded_image = self.encoder(x)
        # x = x.cuda(0)
        # encoded_image = encoded_image.cuda(0)
        # # encode the image with channel = 64
        encoded_noise = self.encoder_mlp(x)
        encoded_positive = self.encoder_mlp(positive)
        encoded_negative = self.encoder_mlp(negative)
        # cross_attention_feature, mu, logvar = self.cross_attention_layer(
        #     encoded_image, encoded_noise)

        # cross_attention_feature = cross_attention_feature.view(
        #     -1, self.dim, self.shape[0], self.shape[1])
        # element wise addition of the noise feature and the cross attention feature

        # encoded_noise_attention = torch.mul(
        #     encoded_noise, cross_attention_feature)

        # encoded_image = torch.cat(
        #     (encoded_noise_attention, encoded_image), dim=1)
        # encoded_image = torch.cat((encoded_noise, encoded_image), dim=1)

        # decoded_image = self.decoder(encoded_image)

        # self.base_model = self.base_model.cuda(1)
        # decoded_image = decoded_image.cuda(1)
        # reference = reference.cuda(1)

        # resnet_out = self.base_model(decoded_image)
        # resnet_out_encoded = self.base_model(reference)
        return encoded_noise, encoded_positive, encoded_negative
    
def prepare_data():

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    # Use selected fold for validation
    train_folds = list(set(opt.all_folds) - set([opt.val_fold]))
    validation_fold = opt.val_fold

    # Train datasets
    image_datasets_train_all = {x: dataset(os.path.join(opt.data_root, x),
                                           os.path.join(opt.ref_root, x),
                                           opt.pkl_root,
                                           transform=data_transforms["train"])
                                for x in train_folds}

    # create one dataset from all datasets of training
    dataset_train = torch.utils.data.ConcatDataset(
        [image_datasets_train_all[i] for i in train_folds])

    # Validation datasets
    dataset_val = dataset(os.path.join(opt.data_root, validation_fold),
                          os.path.join(opt.ref_root, validation_fold),
                          opt.pkl_root,
                          transform=data_transforms["validation"])

    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=opt.bs,
                                                   shuffle=True,
                                                   num_workers=opt.num_workers)

    dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                 batch_size=opt.bs,
                                                 shuffle=False,
                                                 num_workers=opt.num_workers)

    train_size = len(dataset_train)
    val_size = len(dataset_val)

    print("train dataset size =", train_size)
    print("validation dataset size=", val_size)

    return {"train": dataloader_train,
            "val": dataloader_val,
            "dataset_size": {"train": train_size,
                             "val": val_size}}

def train_cifar(config, data_dir=None):
    net = Net(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        net.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0
        
    print("Starting training from epoch========================================================================================", start_epoch)

    dataloaders = prepare_data()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)
    criterion = nn.CrossEntropyLoss()  # weight=weights
    criterion_ae = nn.MSELoss()
    best_acc=0.0
    start_epoch=0
    # LR shceduler
    # scheduler_lr = lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=opt.lr_sch_factor, patience=opt.lr_sch_patience, verbose=True)
    
    # init triplet loss
    triplet_loss = nn.TripletMarginLoss(margin=config["batch_size"], p=2)
    
    net.train()
    trainloader = dataloaders["train"]
    valloader = dataloaders["val"]

    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        print("Epoch {}/{}".format(epoch, 10 - 1))
        running_loss = 0.0
        epoch_steps = 0
        for i, data in tqdm(enumerate(trainloader, 0)):

            inputs, labels, positive, negative, reference, anchor_label = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            encoded_noise, encoded_positive, encoded_negative = net(
                inputs, positive, negative
            )
            loss = triplet_loss(encoded_noise, encoded_positive, encoded_negative)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

    #     # Validation loss
    #     val_loss = 0.0
    #     val_steps = 0
    #     total = 0
    #     correct = 0
    #     for i, data in enumerate(valloader, 0):
    #         with torch.no_grad():
    #             inputs, labels = data
    #             inputs, labels = inputs.to(device), labels.to(device)

    #             outputs = net(inputs)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()

    #             loss = criterion(outputs, labels)
    #             val_loss += loss.cpu().numpy()
    #             val_steps += 1

    #     checkpoint_data = {
    #         "epoch": epoch,
    #         "net_state_dict": net.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #     }
    #     checkpoint = Checkpoint.from_dict(checkpoint_data)

    #     session.report(
    #         {"loss": val_loss / val_steps, "accuracy": correct / total},
    #         checkpoint=checkpoint,
    #     )
    # print("Finished Training")
    

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = os.path.abspath("./data")
    load_data(data_dir)
    config = {
        "l1": tune.choice([2**i for i in range(9)]),
        "l2": tune.choice([2**i for i in range(9)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()

    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))

def test_accuracy(net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total
if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)