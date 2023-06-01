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
from torchvision import transforms
import cv2
# import TripletLoss as TripletLoss
# from sklearn.manifold import TSNE
# from tsnecuda import TSNE
# import matplotlib.pyplot as plt
import os
import copy
# import pandas as pd
# import numpy as np
from torch.utils.tensorboard import SummaryWriter

from torchmetrics import StructuralSimilarityIndexMeasure

from tqdm import tqdm
from torchsummary import summary
from torch.autograd import Variable

from utils.Dataloader_with_path_2labels_ref import ImageFolderWithPaths as dataset
from utils.Dataloader_with_path_Pytorch import ImageFolderWithPaths as datasetfortest

# import string
from networks import MyNet

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
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'validation': transforms.Compose([
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
    dataset_test = datasetfortest(os.path.join(opt.data_root, validation_fold),
                                  transform=data_transforms["validation"])
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=opt.num_workers)

    train_size = len(dataset_train)
    val_size = len(dataset_val)
    test_size = len(dataset_test)

    print("train dataset size =", train_size)
    print("validation dataset size=", val_size)
    print("test dataset size=", test_size)

    return {"train": dataloader_train,
            "val": dataloader_val,
            "test": dataloader_test,
            "dataset_size": {"train": train_size,
                             "val": val_size,
                             "test": test_size}}


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
    triplet_loss = nn.TripletMarginLoss(margin=5.0, p=2)

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
            ssim = StructuralSimilarityIndexMeasure(
                data_range=2.0).to('cuda:2')
            num_batches = 0

            for i, data in tqdm(enumerate(dataloader, 0)):

                inputs, labels, positive, negative, reference, anchor_label = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                reference = reference.to(device)
                anchor_label = anchor_label.to("cuda:2")

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    (resnet_out,
                     resnet_out_encoded,
                     decoded_image,
                     encoded_noise,
                     encoded_positive,
                     encoded_negative,
                     mu,
                     logvar) = model(inputs, positive, negative, reference)

                    # put all data to cpu
                    resnet_out = resnet_out.to('cuda:2')
                    resnet_out_encoded = resnet_out_encoded.to('cuda:2')
                    decoded_image = decoded_image.to('cuda:2')
                    encoded_noise = encoded_noise.to('cuda:2')
                    encoded_negative = encoded_negative.to('cuda:2')
                    encoded_positive = encoded_positive.to('cuda:2')
                    reference = reference.to('cuda:2')

                    loss_feature = criterion_ae(resnet_out, resnet_out_encoded)
                    loss_ae = criterion_ae(decoded_image, reference)
                    loss_triplet = triplet_loss(
                        encoded_noise, encoded_positive, encoded_negative)
                    loss_KL = 0.5 * \
                        torch.sum(mu ** 2 + torch.exp(logvar) - logvar - 1)

                    loss = loss_ae + loss_triplet + loss_feature + loss_KL
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # calculate the PSNR between the original and the decoded image using the MSE of pytorch

                mse += F.mse_loss(decoded_image, reference)
                ssim_batch_tensor = ssim(decoded_image, reference)
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
                if epoch_ssim > best_acc:
                    best_acc = epoch_ssim
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    best_epoch_loss = epoch_loss
                    best_epoch_psnr = epoch_psnr
                    best_epoch_ssim = epoch_ssim
                    best_epoch_mse = epoch_mse
                    print("Found a better model")
                if epoch % 10 == 0:
                    save_model(best_model_wts, best_epoch, best_epoch_loss,
                               best_epoch_psnr, best_epoch_ssim, best_epoch_mse)

                # Get current lr
                lr = optimizer.param_groups[0]['lr']
                # print("lr=", lr)
                writer.add_scalar("LR", lr, epoch)
                scheduler.step(epoch_loss)

            # Print output
            print('Epoch:\t  %d |Phase: \t %s | Loss:\t\t %.4f | PSNR:\t %.4f | MSE:\t %.4f | SSIM:\t %.4f'
                  % (epoch, phase, epoch_loss, epoch_psnr, epoch_mse, epoch_ssim))


# ===============================================
# Prepare models
# ===============================================

def prepare_model(training=True):
    model = MyNet(training=training)
    # model = nn.DataParallel(model, device_ids=[opt.device_id])
    model = model.to(device)
    return model


# ====================================
# Run training process
# ====================================
def run_train(retrain=False):
    torch.cuda.empty_cache()
    model = prepare_model(training=True)
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
        train_model(model, optimizer, criterion, criterion_ae, dataloaders,
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
def check_model_graph():
    model = prepare_model(training=True)
    summary(model, (3, 336, 336))  # this run on GPU
    model = model.to('cpu')
    print(model)
    dummy_input = Variable(torch.rand(13, 3, 336, 336))
    writer.add_graph(model, dummy_input)  # this need the model on CPU

# ===============================================
#  Model testing method
# ===============================================


def test_model():
    print("hint: ./output/TCFA.py/checkpoints/TCFA_epoch:#.pt")
    # test_model_checkpoint = input("Please enter the path of test model:")
    test_model_checkpoint = "./output/TCFA.py/checkpoints/TCFA_epoch:0.pt"
    checkpoint = torch.load(test_model_checkpoint)

    model = prepare_model(training=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataloaders = prepare_data()
    test_dataloader = dataloaders["test"]

    with torch.no_grad():

        for i, data in tqdm(enumerate(test_dataloader, 0)):
            # GPUtil.showUtilization()
            inputs, basename = data
            inputs = inputs.to(device)
            decoded_image = model(inputs, inputs, inputs, inputs)
            # save images to results folder
            save_img("./output/imgresults/" +
                     basename[0], decoded_image[0].cpu().numpy())

    print("Finished testing")


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
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
    # Finish tensorboard writer
    writer.close()

    # show the user that the script has finished running
    print("=====================================")
    print("Finished running script")
