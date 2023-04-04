#  # Developer: Tan Sy NGUYEN
#  # Last modified date: TODAY 2022-12-06
#  # ##################################

#  # Description ##################
#  # pythroch resnet18 training

# Use the attention model to train the model

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
import matplotlib.pyplot as plt
import os
import copy
import pandas as pd
import numpy as np
import itertools
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum
from torch.utils.tensorboard import SummaryWriter

from torchmetrics import StructuralSimilarityIndexMeasure

from tqdm import tqdm
from torchsummary import summary
from torch.autograd import Variable

from Dataloader_with_path_2labels_ref import ImageFolderWithPaths as dataset

import string

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
parser.add_argument("action",
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

# ==========================================
# Device handling
# ==========================================
torch.cuda.set_device(opt.device_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================
# Folder handling
# ===========================================

# make output folder if not exist
os.makedirs(opt.out_dir, exist_ok=True)


# make subfolder in the output folder
# Get python file name (soruce code name)
py_file_name = opt.py_file.split("/")[-1]
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

def train_model(model, optimizer, criterion_ssim, criterion_ae, dataloaders: dict, scheduler, best_acc=0.0, start_epoch=0):

    # best_model_wts = copy.deepcopy(model.state_dict())
    # init triplet loss
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    for epoch in tqdm(range(start_epoch, start_epoch + opt.num_epochs)):

        for phase in ["train", "val"]:

            if phase == "train":
                model.train()
                dataloader = dataloaders["train"]
            else:
                model.eval()
                dataloader = dataloaders["val"]

            running_loss = 0.0
            mse = 0.0
            ssim_batch = 0.0
            ssim = StructuralSimilarityIndexMeasure(data_range=2.0)
            num_batches = 0

            for i, data in tqdm(enumerate(dataloader, 0)):

                inputs, labels, positive, negative, reference = data
                # input_view = inputs.view(inputs.size(0), -1)
                inputs = inputs.to(device)
                labels = labels.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                reference = reference.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    resnet_out, resnet_out_encoded, decoded_image, encoded_noise, encoded_positive, encoded_negative = model(
                        inputs, positive, negative, reference)

                    # put all data to cpu
                    resnet_out = resnet_out.cpu()
                    resnet_out_encoded = resnet_out_encoded.cpu()
                    decoded_image = decoded_image.cpu()
                    encoded_noise = encoded_noise.cpu()
                    encoded_negative = encoded_negative.cpu()
                    encoded_positive = encoded_positive.cpu()
                    # mu = mu.cpu()
                    # logvar = logvar.cpu()
                    reference = reference.cpu()

                    loss_feature = criterion_ae(resnet_out, resnet_out_encoded)
                    loss_ae = criterion_ae(decoded_image, reference)
                    loss_triplet = triplet_loss(
                        encoded_positive, encoded_negative, encoded_noise)
                    loss = loss_ae + loss_triplet + loss_feature
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # torch.cuda.empty_cache()  # decrease about 2GB
                        # del loss  # nothing change

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # calculate the PSNR between the original and the decoded image using the MSE of pytorch

                mse += F.mse_loss(decoded_image, reference)
                ssim_batch_tensor = ssim(decoded_image, reference)
                # print(ssim_batch_tensor.item())
                # print(ssim_batch)
                ssim_batch += ssim_batch_tensor.item()
                # empty cache
                num_batches += 1
                torch.cuda.empty_cache()
            epoch_loss = running_loss / dataloaders["dataset_size"][phase]
            epoch_mse = mse / dataloaders["dataset_size"][phase]
            epoch_psnr = 10 * torch.log10(1 / epoch_mse)
            # calculate SSIM
            epoch_ssim = ssim_batch / num_batches

            # update tensorboard writer
            writer.add_scalars("Loss", {phase: epoch_loss}, epoch)
            writer.add_scalars("PSNR", {phase: epoch_psnr}, epoch)
            writer.add_scalars("mse", {phase: epoch_mse}, epoch)

            # update the lr based on the epoch loss
            if phase == "val":

                # keep best model weights
                if epoch_psnr > best_acc:
                    best_acc = epoch_psnr
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    best_epoch_loss = epoch_loss
                    best_epoch_psnr = epoch_psnr
                    best_epoch_mse = epoch_mse
                    print("Found a better model")

                # Get current lr
                lr = optimizer.param_groups[0]['lr']
                # print("lr=", lr)
                writer.add_scalar("LR", lr, epoch)
                scheduler.step(epoch_loss)

            # Print output
            print('Epoch:\t  %d |Phase: \t %s | Loss:\t\t %.4f | PSNR:\t %.4f | MSE:\t %.4f | SSIM:\t %.4f'
                  % (epoch, phase, epoch_loss, epoch_psnr, epoch_mse, epoch_ssim))

    save_model(best_model_wts, best_epoch, best_epoch_loss,
               best_epoch_psnr, best_epoch_mse)


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


class CrossAttention(nn.Module):
    def __init__(self, dim=2048, heads=8, dim_head=256, dropout=0., patch_size_large=16, input_size=224, channels=128):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        num_patches_large = (input_size // patch_size_large)  # 14

        self.to_patch_embedding_x = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size_large, p2=patch_size_large),  # 14x14 patches with c = 128
            nn.Linear(channels*patch_size_large*patch_size_large, dim),
        )

        self.to_patch_embedding_noise = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size_large, p2=patch_size_large),  # 14x14 patches with c = 128
            nn.Linear(channels*patch_size_large*patch_size_large, dim),
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, channels*patch_size_large*patch_size_large),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                      p1=patch_size_large, p2=patch_size_large, c=channels, h=num_patches_large),  # 14x14 patches with c = 128
        )

    def forward(self, x_q, x_kv):
        x_q = self.to_patch_embedding_x(x_q)
        x_kv = self.to_patch_embedding_noise(x_kv)

        b, n, _, h = *x_q.shape, self.heads

        k = self.to_k(x_kv)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h, b=b, n=n)

        v = self.to_v(x_kv)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h, b=b, n=n)

        q = self.to_q(x_q)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h, b=b, n=n)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.mlp_head(out)
        return out


# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn

#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)


class MyNet(nn.Module):
    def __init__(self, num_out=14):
        super(MyNet, self).__init__()
        self.shape = (224, 224)
        self.dim = 64
        self.attention_dim = 112
        self.flatten_dim = self.shape[0] * self.shape[1] * self.dim

        self.base_model = BaseNet(num_out)
        for param in self.base_model.parameters():
            param.requires_grad = False

        # create an autoencoder block for input size 224*224*3 containing 2 conv layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.cross_attention_layer = CrossAttention(
            dim=self.attention_dim, heads=7, dim_head=32, dropout=0., patch_size_large=16, input_size=224, channels=64)
        self.fc_mu = nn.Linear(self.flatten_dim, 128)
        self.fc_var = nn.Linear(self.flatten_dim, 128)

        # MLP to encode the image to extract the noise level
        self.encoder_mlp = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.decoder_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Linear(512, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(),
            nn.Linear(256, self.flatten_dim),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        std = std.to(device)
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        esp = esp.to(device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x, positive, negative, reference):
        x = x.cuda(1)
        self.encoder = self.encoder.cuda(1)
        encoded_image = self.encoder(x)
        x = x.cuda(0)
        encoded_image = encoded_image.cuda(0)
        # encode the image with channel = 64
        encoded_noise = self.encoder_mlp(x)
        encoded_positive = self.encoder_mlp(positive)
        encoded_negative = self.encoder_mlp(negative)

        # encoded_noise_flatten = encoded_noise.view(-1, self.flatten_dim)
        # z, mu, logvar = self.bottleneck(encoded_noise_flatten)
        # noise_feature = self.decoder_mlp(z)
        # noise_feature = noise_feature.view(-1,
        #                                    self.dim, self.shape[0], self.shape[1])
        cross_attention_feature = self.cross_attention_layer(
            encoded_image, encoded_noise)

        # mu = 0
        # logvar = 0

        cross_attention_feature = cross_attention_feature.view(
            -1, self.dim, self.shape[0], self.shape[1])
        # element wise addition of the noise feature and the cross attention feature

        encoded_noise_attention = torch.mul(encoded_noise, cross_attention_feature)

        encoded_image = torch.cat((encoded_noise_attention, encoded_image), dim=1)
        # encoded_image = torch.cat((encoded_noise, encoded_image), dim=1)

        decoded_image = self.decoder(encoded_image)

        self.base_model = self.base_model.cuda(1)
        decoded_image = decoded_image.cuda(1)
        reference = reference.cuda(1)

        resnet_out = self.base_model(decoded_image)
        resnet_out_encoded = self.base_model(reference)
        return resnet_out, resnet_out_encoded, decoded_image, encoded_noise, encoded_positive, encoded_negative


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
    torch.cuda.empty_cache()
    model = prepare_model()

    dataloaders = prepare_data()

    # optimizer = optim.Adam(model.parameters(), lr=opt.lr , weight_decay=opt.weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr )

    # criterion =  nn.MSELoss() # backprop loss calculation
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
        train_model(model, optimizer, criterion, criterion_ae, dataloaders,
                    scheduler, best_acc=0.0, start_epoch=0)


# =====================================
# Save models
# =====================================
def save_model(model_weights,  best_epoch,  best_epoch_loss, best_epoch_psnr, best_epoch_mse):

    # get code file name and make a name
    check_point_name = py_file_name + "_epoch:{}.pt".format(best_epoch)
    check_point_path = os.path.join(checkpoint_dir, check_point_name)
    # save torch model
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": model_weights,
        # "optimizer_state_dict": optimizer.state_dict(),
        # "train_loss": train_loss,
        "loss": best_epoch_loss,
        "psnr": best_epoch_psnr,
        "mse": best_epoch_mse,
    }, check_point_path)


# =====================================
# Check model
# =====================================
def check_model_graph():
    model = prepare_model()
    summary(model, (3, 224, 224))  # this run on GPU
    model = model.to('cpu')
    print(model)
    dummy_input = Variable(torch.rand(13, 3, 224, 224))
    writer.add_graph(model, dummy_input)  # this need the model on CPU

# ===============================================
#  Model testing method
# ===============================================


def test_model():

    test_model_checkpoint = input("Please enter the path of test model:")
    checkpoint = torch.load(test_model_checkpoint)

    model = prepare_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    total = 0
    running_mse = 0.0

    dataloaders = prepare_data()
    test_dataloader = dataloaders["val"]

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader, 0)):

            inputs, labels, positive, negative, reference, noise_level = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            reference = reference.to(device)
            noise_level = noise_level.to(device)
            total += labels.size(0)

            _, _, decoded_image, _, _, _, _, _ = model(
                inputs, positive, negative, reference, positive.size(0))
            mse = F.mse_loss(decoded_image, reference)
            # calculate the mse of the decoded image regarding to each noise level
            mse_list = []

            for noise_level_i in np.unique(noise_level):
                noise_level_i = noise_level_i.to(device)
                mse_i = F.mse_loss(
                    decoded_image[noise_level == noise_level_i], reference[noise_level == noise_level_i])
                print("MSE of noise level %d is %f" % (noise_level_i, mse_i))
                mse_list.append(zip(noise_level_i, mse_i))

            running_mse += mse.item()

    print('copying some data back to cpu for generating confusion matrix...')
    running_mse = running_mse.cpu()

    print('Accuracy of the network on the %d test images: %f %%' %
          (total, (running_mse / total)))

    psnr = 10 * torch.log10(total / running_mse)
    psnr = psnr.cpu()  # to('cpu')

    print('Finished.. ')

    # ====================================================================
    # Writing to a file
    # =====================================================================

    np.set_printoptions(linewidth=np.inf)
    with open("%s/%s_evaluation.csv" % (opt.out_dir, py_file_name), "w") as f:

        # write the psnr and mse of each noise level in the mse_list
        f.write("PSNR: %f\n" % psnr)
        for noise_level_i, mse_i in mse_list:
            f.write("MSE of noise level %d is %f\n" % (noise_level_i, mse_i))

    f.close()
    print("Report generated")


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

            # print("paths:", paths)
            filename = [list(paths)[0].split("/")[-1]]
            # print("filenames:", filename)

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

def prepare_submission_file(image_names, predicted_labels, max_probability, time_per_image, submit_dir, data_classes):

    predicted_label_names = []

    for i in predicted_labels:
        predicted_label_names = predicted_label_names + [data_classes[i]]

    #  print(predicted_label_names)

    submission_dataframe = pd.DataFrame(np.column_stack([image_names,
                                                         predicted_label_names,
                                                         max_probability,
                                                         time_per_image]),
                                        columns=['images', 'labels', 'PROB', 'time'])
    # print("image names:{0}".format(image_names))

    submission_dataframe.to_csv(os.path.join(
        submit_dir, "method_3_test_output"), index=False)

    print(submission_dataframe)
    print("successfully created submission file")


# ==============================================
#  Ploting history and save plots to plots directory
# ==============================================


# ==============================================
# Plot confusion matrix - method
# ==============================================
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          plt_size=[15, 12]):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.rcParams['figure.figsize'] = plt_size
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    LABEL_TO_LETTER = {
        "Ampulla of vater": "A",
        "Angiectasia": "B",
        "Blood - fresh": "C",
        "Blood - hematin": "D",
        "Erosion": "E",
        "Erythema": "F",
        "Foreign body": "G",
        "Ileocecal valve": "H",
        "Lymphangiectasia": "I",
        "Normal clean mucosa": "J",
        "Polyp": "K",
        "Pylorus": "L",
        "Reduced mucosal view": "M",
        "Ulcer": "N"
    }
    class_str = [LABEL_TO_LETTER[i] for i in classes]
    plt.xticks(tick_marks, class_str, rotation=90)
    plt.yticks(tick_marks, class_str)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig_path = "%s/%s_matrix.png" % (opt.out_dir, py_file_name)
    plt.savefig(fig_path)
    figure = plt.gcf()
    writer.add_figure("Confusion Matrix", figure)
    print("Finished confusion matrix drawing...")


# ========================================
# Doing Inference for new data
# =========================================


def inference():
    test_model_checkpoint = input("Please enter the path of test model:")
    checkpoint = torch.load(test_model_checkpoint)

    model = prepare_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    trnsfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset_new = dataset(opt.data_to_inference, trnsfm)
    dataloader_new = torch.utils.data.DataLoader(dataset_new,
                                                 batch_size=opt.bs,
                                                 shuffle=False,
                                                 num_workers=opt.num_workers)

    class_names = list(string.ascii_uppercase)[:14]
    print(class_names)
    print("lenth of dataloader:", len(dataloader_new))
    df = pd.DataFrame(columns=["filename", "predicted-label"] + class_names)

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader_new, 0)):

            inputs, labels, paths = data

            df_temp = pd.DataFrame(
                columns=["filename", "predicted-label"] + class_names)

            # print("paths:", paths)
            filenames = []
            for p in paths:
                filenames = filenames + [list(p.split("/"))[-1]]

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = F.softmax(outputs, 1)

            predicted_probability, predicted = torch.max(outputs.data, 1)
            predicted = predicted.data.cpu().numpy()

            df_temp["predicted-label"] = predicted
            df_temp["filename"] = filenames

            probabilities = outputs.cpu().squeeze()
            probabilities = probabilities.tolist()
            probabilities = np.around(probabilities, decimals=3)

            df_temp[class_names] = probabilities
            df = df.append(df_temp)
            # break

    print(df.head())
    print("length of DF:", len(df))
    prob_file_name = "%s/%s_inference.csv" % (opt.out_dir, py_file_name)
    df.to_csv(prob_file_name, index=False)

# ======================================
# Main function to run the code
# ======================================


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
    elif opt.action == "test":
        print("=====================================")
        print("Inference process is started..!")
        test_model()
    elif opt.action == "check":
        print("=====================================")
        check_model_graph()
        print("Check pass")
    elif opt.action == "prepare":
        print("=====================================")
        prepare_prediction_file()
        print("Probability file prepared..!")
    elif opt.action == "inference":
        print("=====================================")
        inference()
        print("Inference completed")

    # Finish tensorboard writer
    writer.close()

    # show the user that the script has finished running
    print("=====================================")
    print("Finished running script")
