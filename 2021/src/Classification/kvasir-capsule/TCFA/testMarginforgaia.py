#  # Developer: Tan Sy NGUYEN
#  # Last modified date: TODAY 2022-12-06
#  # ##################################

#  # Description ##################
#  # pythroch resnet18 training
#  Use the attention with mu and logvar to train the model

###########################################

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
# import TripletLoss as TripletLoss
# import gc
import os
# import copy
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# from torchmetrics import StructuralSimilarityIndexMeasure

from tqdm import tqdm
# from torchsummary import summary
# from torch.autograd import Variable

from utils.dataloaderforgaia import ImageFolderWithPaths as dataset

# inport davies bouldin index
from sklearn.metrics import davies_bouldin_score
# import silhouette score
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

# print all input parameters to the console
print(opt)

# ==========================================
# Device handling
# ==========================================
# torch.cuda.set_device(opt.device_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print("GPU is available")
    # print name of the GPU and the number of available GPUs

    print("Device name: ", torch.cuda.get_device_name(0))
    print("Number of GPUs: ", torch.cuda.device_count())

# ===========================================
# Folder handling
# ===========================================

# make output folder if not exist
os.makedirs(opt.out_dir, exist_ok=True)
os.makedirs(opt.mat_dir, exist_ok=True)


# make subfolder in the output folder
# Get python file name (soruce code name)
py_file_name = opt.py_file.split("/")[-1]
basename = os.path.splitext(py_file_name)[0]
checkpoint_dir = os.path.join(opt.out_dir, py_file_name + "/checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# make tensorboard subdirectory for the experiment
tensorboard_exp_dir = os.path.join(opt.tensorboard_dir, py_file_name)
os.makedirs(tensorboard_exp_dir, exist_ok=True)


# ==========================================
# Tensorboard
# ==========================================
# Initialize summary writer
writer = SummaryWriter(tensorboard_exp_dir)


# ==========================================
# Prepare Data
# ==========================================
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


# ==========================================================
# Train model
# ==========================================================

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def train_model(model,
                optimizer,
                criterion_ssim,
                criterion_ae,
                dataloaders: dict,
                scheduler,
                best_acc=0.0,
                start_epoch=0):

    # best_model_wts = copy.deepcopy(model.state_dict())
    # init triplet loss
    # for margin in tqdm([1.0, 2, 4, 8, 16, 32, 64]):
    triplet_loss = nn.TripletMarginLoss(margin=criterion_ssim, p=2)

    for epoch in tqdm(range(start_epoch, start_epoch + opt.num_epochs)):
        # create two empty list to store the features and labels with array two dimensions

        tsne_features_in = np.empty((19980, 193600))
        tsne_labels_in = np.empty((19980, 1))

        # tsne_features_in = torch.Tensor(tsne_features).to(device)
        # tsne_labels_in = torch.Tensor(tsne_labels).to(device)

        for phase in ["train", "val"]:

            if phase == "train":
                model.train()
                dataloader = dataloaders["train"]
            else:
                model.eval()
                dataloader = dataloaders["val"]

            running_loss = 0.0
            # mse = 0.0
            # ssim_batch = 0.0
            # ssim = StructuralSimilarityIndexMeasure(
            #     data_range=2.0).to(device)
            # num_batches = 0

            for i, data in tqdm(enumerate(dataloader, 0)):

                inputs, labels, positive, negative, anchor_label = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                # reference = reference.to(device)
                anchor_label = anchor_label.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    encoded_noise, encoded_positive, encoded_negative = model(
                        inputs, positive, negative)

                    encoded_noise = encoded_noise.to(device)
                    encoded_positive = encoded_positive.to(device)
                    encoded_negative = encoded_negative.to(device)

                    num_elements = encoded_noise.shape[0]

                    loss_triplet = triplet_loss(
                        encoded_noise, encoded_positive, encoded_negative)

                    # flatten the tensors encoded_noise over the batch dimension
                    encoded_noise_flatten = encoded_noise.view(
                        encoded_noise.shape[0], -1)

                    # append the encoded_noise_flatten to the tsne_features to the new line
                    if phase == 'val':
                        if num_elements == opt.bs:
                            tsne_features_in[i * opt.bs: (
                                i + 1) * opt.bs, :] = encoded_noise_flatten.cpu().detach().numpy()
                            tsne_labels_in[i * opt.bs: (i + 1) * opt.bs,
                                            :] = anchor_label.cpu().detach().numpy()
                        else:
                            tsne_features_in[i * opt.bs: i * opt.bs + num_elements,
                                                :] = encoded_noise_flatten.cpu().detach().numpy()
                            tsne_labels_in[i * opt.bs: i * opt.bs + num_elements,
                                            :] = anchor_label.cpu().detach().numpy()

                    loss = loss_triplet
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                # break

                # num_batches += 1
                torch.cuda.empty_cache()

            # calculate the davies bouldin score
            if phase == 'val':
                print("Calculating the davies bouldin score")
                # print(tsne_features_in.shape)
                # tsne_features_cpu = np.reshape(
                #     tsne_features_in, (-1, 193600))

                # tsne_labels_cpu = np.reshape(tsne_labels_in, (-1, 1))
                tsne_features_cpu = np.squeeze(tsne_features_in)
                tsne_labels_cpu = np.squeeze(tsne_labels_in)

                tsne_features_cpu = np.squeeze(tsne_features_cpu)
                tsne_labels_cpu = np.squeeze(tsne_labels_cpu)

                score_davies = davies_bouldin_score(
                    tsne_features_cpu, tsne_labels_cpu)

                # calculate the silhouette score
                score_silhouette = silhouette_score(
                    tsne_features_cpu, tsne_labels_cpu)

                # save epoch and score to the txt file
                with open('margin_{margin}.txt'.format(margin=criterion_ssim), 'a') as f:
                    f.write("Epoch: %d, Davies Bouldin score: %.4f, Silhouette score: %.4f\n" % (
                        epoch, score_davies, score_silhouette))

# ================================================
# New architecture
# ================================================


class BaseNet(nn.Module):
    def __init__(self, num_out=14):
        super(BaseNet, self).__init__()
        self.resnet_model = models.densenet121(pretrained=True)
        self.module = nn.Sequential(*list(self.resnet_model.children())[:-1])

    def forward(self, x):
        x = self.module(x)
        return x


class MyNet(nn.Module):
    def __init__(self, num_out=14):
        super(MyNet, self).__init__()
        self.shape = (224, 224)
        self.dim = 64
        self.attention_dim = 112
        self.flatten_dim = self.shape[0] * self.shape[1] * self.dim

        self.encoder_mlp = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, x, positive, negative):
        x = x.cuda(0)
        encoded_noise = self.encoder_mlp(x)
        encoded_positive = self.encoder_mlp(positive)
        encoded_negative = self.encoder_mlp(negative)
        return encoded_noise, encoded_positive, encoded_negative


# ===============================================
# Prepare models
# ===============================================

def prepare_model():
    model = MyNet()
    # model = nn.DataParallel(model, device_ids=[opt.device_id])
    model = model.to(device)
    return model


# ====================================
# Run training process
# ====================================
def run_train(retrain=False):
    for margin in tqdm([2, 4, 8, 16, 32, 64]):
        torch.cuda.empty_cache()
        model = prepare_model()
        dataloaders = prepare_data()
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()  # weight=weights
        criterion_ae = nn.MSELoss()
        # LR shceduler
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=opt.lr_sch_factor, patience=opt.lr_sch_patience, verbose=True)
        # call main train loop
        if retrain:
            # train from a checkpoint
            checkpoint_path = input("Please enter the checkpoint path:")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint["epoch"]
            # loss = checkpoint["loss"]
            acc = checkpoint["psnr"]
            train_model(model, optimizer, criterion, criterion_ae, dataloaders,
                        scheduler, best_acc=acc, start_epoch=start_epoch)

        else:
            train_model(model, optimizer, margin, criterion_ae, dataloaders,
                        scheduler, best_acc=0.0, start_epoch=0)


# =====================================
# Save models
# =====================================
def save_model(model_weights,  best_epoch,  best_epoch_loss, best_epoch_psnr, best_epoch_ssim, best_epoch_mse):

    # get code file name and make a name
    check_point_name = basename + "_epoch:{}.pt".format(best_epoch)
    check_point_path = os.path.join(checkpoint_dir, check_point_name)
    # save torch model
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": model_weights,
        "loss": best_epoch_loss,
        "psnr": best_epoch_psnr,
        "mse": best_epoch_mse,
        "ssim": best_epoch_ssim,
    }, check_point_path)


# =====================================
# Check model
# =====================================
# def check_model_graph():
#     model = prepare_model()
#     summary(model, (3, 224, 224))  # this run on GPU
#     model = model.to('cpu')
#     print(model)
#     dummy_input = Variable(torch.rand(13, 3, 224, 224))
#     writer.add_graph(model, dummy_input)  # this need the model on CPU


# ==============================================
# Prepare submission file with probabilities
# ===============================================


def prepare_prediction_file():

    if opt.bs != 1:
        print("Please run with bs = 1")
        exit()

    test_model_checkpoint = input("Please enter the path of test model:")
    checkpoint = torch.load(test_model_checkpoint)

    model = prepare_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataloaders = prepare_data()
    test_dataloader = dataloaders["val"]

    class_names = test_dataloader.dataset.classes

    df = pd.DataFrame(
        columns=["filename", "predicted-label", "actual-label"] + class_names)

    print(df.head())

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader, 0)):

            inputs, labels, paths = data

            df_temp = pd.DataFrame(
                columns=["filename", "predicted-label", "actual-label"] + class_names)
            filename = [list(paths)[0].split("/")[-1]]
            df_temp["filename"] = filename

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = F.softmax(outputs, 1)
            predicted_probability, predicted = torch.max(outputs.data, 1)

            df_temp["predicted-label"] = class_names[predicted.item()]
            df_temp["actual-label"] = class_names[labels.item()]

            probabilities = outputs.cpu().squeeze()
            probabilities = probabilities.tolist()
            probabilities = np.around(probabilities, decimals=3)
            # print(probabilities)

            df_temp[class_names] = probabilities
            df = df.append(df_temp)

        print(df.head())
        print("length of DF:", len(df))
        prob_file_name = "%s/%s_probabilities.csv" % (
            opt.out_dir, py_file_name)
        df.to_csv(prob_file_name, index=False)


# ==============================================
# Prepare submission file:
# ===============================================


if __name__ == '__main__':
    print("Started data preparation")
    data_loaders = prepare_data()
    # print(vars(opt))
    print("=====================================")
    print("Data is ready")

    # Train or retrain or inference
    if opt.action == "train":
        print("=====================================")
        print("Training process is started..!")
        run_train()
        # pass
    elif opt.action == "retrain":
        print("=====================================")
        print("Retrainning process is started..!")
        run_train(retrain=True)
        # pass

    # Finish tensorboard writer
    writer.close()

    # show the user that the script has finished running
    print("=====================================")
    print("Finished running script")
