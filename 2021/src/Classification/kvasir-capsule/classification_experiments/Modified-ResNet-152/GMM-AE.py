#  # Developer: Vajira Thambawita
#  # Last modified date: 18/07/2018
#  # ##################################

#  # Description ##################
#  # pythroch resnet18 training


###########################################

from __future__ import print_function, division


# #start = datetime.datetime.now()
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
import TripletLoss as TripletLoss
# from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt
import os
import copy
import pandas as pd
import numpy as np

import sklearn.metrics as mtc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from torchsummary import summary
from torch.autograd import Variable

from Dataloader_with_path_2labels import ImageFolderWithPaths as dataset

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
                    default="dataport/ExperimentalDATA/ui/",
                    help="data root directory")

parser.add_argument("--pkl_root",
                    default="dataport/ExperimentalDATA/ui/",
                    help="pkl root directory")


parser.add_argument("--data_to_inference",
                    default="dataport/interference/clean-ui/",
                    help="Data folder with one subfolder which containes images to do inference")

parser.add_argument("--out_dir",
                    default="dataport/output/clean-ui/",
                    help="Main output dierectory")

parser.add_argument("--tensorboard_dir",
                    default="dataport/tensorboard/clean-ui/",
                    help="Folder to save output of tensorboard")

# Hyper parameters
parser.add_argument("--bs", type=int, default=32, help="Mini batch size")
parser.add_argument("--lr", type=float, default=0.001,
                    help="Learning rate for training")
parser.add_argument("--num_workers", type=int, default=32,
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
parser.add_argument("--num_epochs", type=int, default=0,
                    help="Numbe of epochs to train")
# parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch in retraining")
parser.add_argument("action", type=str, help="Select an action to run", choices=[
                    "train", "retrain", "test", "check", "prepare", "inference"])
parser.add_argument("--checkpoint_interval", type=int,
                    default=25, help="Interval to save checkpoint models")
parser.add_argument("--val_fold", type=str, default="0",
                    help="Select the validation fold", choices=["0", "1"])
parser.add_argument(
    "--all_folds", default=["0", "1"], help="list of all folds available in data folder")

parser.add_argument("--best_resnet", default="/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/output/ref/train-0_val-1/fine-tuned-kvasircapsule.py/checkpoints/fine-tuned-kvasircapsule.py_epoch:48.pt", help="Resnet best weight file")
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


###########################################################
###########################################################
###########################################


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
    image_datasets_train_all = {x: dataset(os.path.join(opt.data_root, x), opt.pkl_root,
                                           transform=data_transforms["train"])
                                for x in train_folds}

    # create one dataset from all datasets of training
    dataset_train = torch.utils.data.ConcatDataset(
        [image_datasets_train_all[i] for i in train_folds])

    # Validation datasets
    dataset_val = dataset(os.path.join(opt.data_root, validation_fold),opt.pkl_root,
                          transform=data_transforms["validation"])

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.bs,
                                                   shuffle=True, num_workers=opt.num_workers)

    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.bs,
                                                 shuffle=False, num_workers=opt.num_workers)

    train_size = len(dataset_train)
    val_size = len(dataset_val)

    print("train dataset size =", train_size)
    print("validation dataset size=", val_size)

    return {"train": dataloader_train, "val": dataloader_val, "dataset_size": {"train": train_size, "val": val_size}}


# ==========================================================
# Train model
# ===========================================================

def train_model(model, optimizer, criterion, criterion_ae, dataloaders: dict, scheduler, best_acc=0.0, start_epoch=0):

    best_model_wts = copy.deepcopy(model.state_dict())
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
            running_corrects = 0

            for i, data in tqdm(enumerate(dataloader, 0)):

                inputs, labels, positive, negative = data
                input_view = inputs.view(inputs.size(0), -1)
                inputs = inputs.to(device)
                labels = labels.to(device)
                positive = positive.view(positive.size(0), -1).to(device)
                negative = negative.view(negative.size(0), -1).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    resnet_out, decoded_image, encoded_noise, encoded_positive, encoded_negative, mu, logvar = model(
                        inputs, input_view, positive, negative, positive.size(0))
                    _, preds = torch.max(resnet_out, 1)
                    loss_resnet = criterion(resnet_out, labels)
                    loss_ae = criterion_ae(decoded_image, inputs)
                    loss_triplet = triplet_loss(
                        encoded_positive, encoded_negative, encoded_noise)
                    loss_KL = 0.5 * \
                        torch.sum(mu ** 2 + torch.exp(logvar) - logvar - 1)

                    # print(loss_resnet, loss_ae, loss_triplet, loss_KL)
                    loss = loss_resnet + loss_ae + loss_triplet + loss_KL

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataloaders["dataset_size"][phase]
            epoch_acc = running_corrects.double(
            ) / dataloaders["dataset_size"][phase]

            # update tensorboard writer
            writer.add_scalars("Loss", {phase: epoch_loss}, epoch)
            writer.add_scalars("Accuracy", {phase: epoch_acc}, epoch)

            # update the lr based on the epoch loss
            if phase == "val":

                # keep best model weights
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    best_epoch_loss = epoch_loss
                    best_epoch_acc = epoch_acc
                    print("Found a better model")

                # Get current lr
                lr = optimizer.param_groups[0]['lr']
                #print("lr=", lr)
                writer.add_scalar("LR", lr, epoch)
                scheduler.step(epoch_loss)

            # Print output
            print('Epoch:\t  %d |Phase: \t %s | Loss:\t\t %.4f | Acc:\t %.4f '
                  % (epoch, phase, epoch_loss, epoch_acc))

    save_model(best_model_wts, best_epoch, best_epoch_loss, best_epoch_acc)


# ================================================
# New architecture
# ================================================


class BaseNet(nn.Module):
    def __init__(self, num_out=14):
        super(BaseNet, self).__init__()
        self.resnet_model = models.resnet152()
        self.resnet_num_ftrs = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Linear(self.resnet_num_ftrs, num_out)

    def forward(self, x):
        x = self.resnet_model(x)
        return x


class MyNet(nn.Module):
    def __init__(self, num_out=14):
        super(MyNet, self).__init__()

        self.base_model = BaseNet(num_out)
        self.base_model = nn.DataParallel(
            self.base_model, device_ids=[opt.device_id])
        checkpoint_resnet = torch.load(opt.best_resnet)
        self.base_model.load_state_dict(
            checkpoint_resnet["model_state_dict"])  # Load best weight
        self.base_model.to(device)
        # freeze all layers
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

        self.fc_mu = nn.Linear(256, 128)
        self.fc_var = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(256, 336*336*3)

        # MLP to encode the image to extract the noise level
        self.encoder_mlp = nn.Sequential(
            nn.Linear(224*224*3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # MLP to decode the image
        self.decoder_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 224*224*3),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
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

    def forward(self, x,x_view, positive, negative, shape):
        encoded_image = self.encoder(x)
        encoded_noise = self.encoder_mlp(x_view)
        encoded_positive = self.encoder_mlp(positive)
        encoded_negative = self.encoder_mlp(negative)

        z, mu, logvar = self.bottleneck(encoded_noise)
        noise = self.decoder_mlp(z)
        noise = noise.view(shape, 3, 224, 224)

        decoded_image = self.decoder(encoded_image)


        # cacade the noise to the decoded image
        decoded_image = decoded_image + noise

        resnet_out = self.base_model(decoded_image)
        return resnet_out, decoded_image, encoded_noise, encoded_positive, encoded_negative, mu, logvar


# ===============================================
# Prepare models
# ===============================================

def prepare_model():

    model = MyNet()
    model = nn.DataParallel(model, device_ids=[opt.device_id])
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
    criterion_ae = nn.MSELoss()  # Absolute error for real loss calculations

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
        loss = checkpoint["loss"]
        acc = checkpoint["acc"]
        train_model(model, optimizer, criterion, criterion_ae, dataloaders,
                    scheduler, best_acc=acc, start_epoch=start_epoch)

    else:
        train_model(model, optimizer, criterion, criterion_ae, dataloaders,
                    scheduler, best_acc=0.0, start_epoch=0)


# =====================================
# Save models
# =====================================
def save_model(model_weights,  best_epoch,  best_epoch_loss, best_epoch_acc):

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
        "acc": best_epoch_acc,
    }, check_point_path)


# =====================================
# Check model
# =====================================
def check_model_graph():
    model = prepare_model()

    summary(model, (3, 224, 224))  # this run on GPU
    model = model.to('cpu')
    #dataloaders = prepare_data()
    #sample = next(iter(dataloaders["train"]))

    #inputs = sample["features"]
   # inputs = inputs.to(device, torch.float)
    # print(inputs.shape)
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

    dataloaders = prepare_data()
    test_dataloader = dataloaders["val"]

    # TO collect data
    correct = 0
    total = 0
    all_labels_d = torch.tensor([], dtype=torch.long).to(device)
    all_predictions_d = torch.tensor([], dtype=torch.long).to(device)
    all_predictions_probabilities_d = torch.tensor(
        [], dtype=torch.float).to(device)

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader, 0)):

            inputs, labels, positive, negative, noise_level = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            noise_level = noise_level.to(device)

            resnet_out, _, _, _, _, _, _ = model(
                        inputs, positive, negative)
            outputs = F.softmax(resnet_out, 1)
            predicted_probability, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum()
            all_labels_d = torch.cat((all_labels_d, labels), 0)
            all_predictions_d = torch.cat((all_predictions_d, predicted), 0)
            all_predictions_probabilities_d = torch.cat(
                (all_predictions_probabilities_d, predicted_probability), 0)
            #all_timePerFrame_host = all_timePerFrame_host + [time_per_image]
            # print("testing")

    print('copying some data back to cpu for generating confusion matrix...')
    y_true = all_labels_d.cpu()
    y_predicted = all_predictions_d.cpu()  # to('cpu')
    testset_predicted_probabilites = all_predictions_probabilities_d.cpu()  # to('cpu')

    # return y_predicted, testset_predicted_probabilites, all_timePerFrame_host

    cm = confusion_matrix(y_true, y_predicted)  # confusion matrix

    print('Accuracy of the network on the %d test images: %f %%' % (total, (
        100.0 * correct / total)))

    print(cm)

    print("taking class names to plot CM")

    # test_datasets.classes  # taking class names for plotting confusion matrix
    class_names = test_dataloader.dataset.classes

    print("Generating confution matrix")

    plot_confusion_matrix(cm, classes=class_names, title='my confusion matrix')

    ##################################################################
    # classification report
    #################################################################
    print(classification_report(y_true, y_predicted, target_names=class_names))

    ##################################################################
    # Standard metrics for medico Task
    #################################################################
    print("Printing standard metric for medico task")

    print("Accuracy =", mtc.accuracy_score(y_true, y_predicted))
    print("Precision score =", mtc.precision_score(
        y_true, y_predicted, average="weighted"))
    print("Recall score =", mtc.recall_score(
        y_true, y_predicted, average="weighted"))
    print("F1 score =", mtc.f1_score(y_true, y_predicted, average="weighted"))
    print("Specificity =")
    print("MCC =", mtc.matthews_corrcoef(y_true, y_predicted))

    ##################################################################
    # Standard metrics for medico Task
    #################################################################
    print("Printing standard metric for medico task")

    print("1. Recall score (REC) =", mtc.recall_score(
        y_true, y_predicted, average="weighted"))
    print("2. Precision score (PREC) =",
          mtc.precision_score(y_true, y_predicted, average="weighted"))
    print("3. Specificity (SPEC) =")
    # print("4. Accuracy (ACC) =", mtc.accuracy_score(y_true, y_predicted, weights))
    print("5. Matthews correlation coefficient(MCC) =",
          mtc.matthews_corrcoef(y_true, y_predicted))

    print("6. F1 score (F1) =", mtc.f1_score(
        y_true, y_predicted, average="weighted"))

    print('Finished.. ')

    # ====================================================================
    # Writing to a file
    # =====================================================================

    np.set_printoptions(linewidth=np.inf)
    with open("%s/%s_evaluation.csv" % (opt.out_dir, py_file_name), "w") as f:

        f.write(np.array2string(mtc.confusion_matrix(
            y_true, y_predicted), separator=", "))

        f.write("\n\n\n\n")
        f.write("--- Macro Averaged Resutls ---\n")
        f.write("Precision: %s\n" % mtc.precision_score(
            y_true, y_predicted, average="macro"))
        f.write("Recall: %s\n" % mtc.recall_score(
            y_true, y_predicted, average="macro"))
        f.write("F1-Score: %s\n\n" %
                mtc.f1_score(y_true, y_predicted, average="macro"))

        f.write("--- Micro Averaged Resutls ---\n")
        f.write("Precision: %s\n" % mtc.precision_score(
            y_true, y_predicted, average="micro"))
        f.write("Recall: %s\n" % mtc.recall_score(
            y_true, y_predicted, average="micro"))
        f.write("F1-Score: %s\n\n" %
                mtc.f1_score(y_true, y_predicted, average="micro"))

        f.write("--- Other Resutls ---\n")
        f.write("MCC: %s\n" % mtc.matthews_corrcoef(y_true, y_predicted))

    f.close()
    print("Report generated")

    # ==========================================================================


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
   #  exit()

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader, 0)):

            inputs, labels, paths = data

            df_temp = pd.DataFrame(
                columns=["filename", "predicted-label", "actual-label"] + class_names)

            #print("paths:", paths)
            filename = [list(paths)[0].split("/")[-1]]
            #print("filenames:", filename)

            df_temp["filename"] = filename

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = F.softmax(outputs, 1)
            predicted_probability, predicted = torch.max(outputs.data, 1)

            df_temp["predicted-label"] = class_names[predicted.item()]
            df_temp["actual-label"] = class_names[labels.item()]

            # print("actual label:", labels.item())
            #print("predicted label:", predicted.item())
            # print("probabilities :", outputs.cpu())

            probabilities = outputs.cpu().squeeze()
            probabilities = probabilities.tolist()
            probabilities = np.around(probabilities, decimals=3)
            # print(probabilities)

            df_temp[class_names] = probabilities

            #record = record + [class_names[labels.item()]] + [class_names[predicted.item()]]

            # print(record)
            # print(df_temp.head())
            df = df.append(df_temp)
           # break

        print(df.head())
        print("length of DF:", len(df))
        prob_file_name = "%s/%s_probabilities.csv" % (
            opt.out_dir, py_file_name)
        df.to_csv(prob_file_name, index=False)


##########################################################
# Prepare submission file:
##########################################################

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
    #print("image names:{0}".format(image_names))

    submission_dataframe.to_csv(os.path.join(
        submit_dir, "method_3_test_output"), index=False)

    print(submission_dataframe)
    print("successfully created submission file")
###########################################################

###########################################################
#  Ploting history and save plots to plots directory
###########################################################


############################################################
# Plot confusion matrix - method
############################################################
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
        "Ampulla of vater": "A", "Angiectasia": "B", "Blood - fresh": "C", "Blood - hematin": "D", "Erosion": "E",
        "Erythema": "F", "Foreign body": "G", "Ileocecal valve": "H", "Lymphangiectasia": "I", "Normal clean mucosa": "J",
        "Polyp": "K", "Pylorus": "L", "Reduced mucosal view": "M", "Ulcer": "N"
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

    # if opt.bs != 1:
    #    print("Please run with bs = 1")
    #   exit()

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
    dataloader_new = torch.utils.data.DataLoader(dataset_new, batch_size=opt.bs,
                                                 shuffle=False, num_workers=opt.num_workers)

    class_names = list(string.ascii_uppercase)[:14]
    print(class_names)
    print("lenth of dataloader:", len(dataloader_new))
    df = pd.DataFrame(columns=["filename", "predicted-label"] + class_names)

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader_new, 0)):

            inputs, labels, paths = data

            df_temp = pd.DataFrame(
                columns=["filename", "predicted-label"] + class_names)

            #print("paths:", paths)
            filenames = []
            for p in paths:
                # print(p)
                # print([list(p.split("/"))[-1]])
                filenames = filenames + [list(p.split("/"))[-1]]
            #print("filenames:", filename)

            #df_temp["filename"] = filename

            #print("file names:", filenames)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = F.softmax(outputs, 1)
            predicted_probability, predicted = torch.max(outputs.data, 1)

            predicted = predicted.data.cpu().numpy()

            #print("predicted items=", predicted)
            #print("paths:", paths)

            df_temp["predicted-label"] = predicted
            df_temp["filename"] = filenames
            # df_temp[2:-1] =

            # print(df_temp)
            #df_temp["actual-label"] = class_names[labels.item()]

            # print("actual label:", labels.item())
            #print("predicted label:", predicted.item())
            # print("probabilities :", outputs.cpu())

            probabilities = outputs.cpu().squeeze()
            probabilities = probabilities.tolist()
            probabilities = np.around(probabilities, decimals=3)
            # print(probabilities)

            # print(probabilities)
            df_temp[class_names] = probabilities

            # print(df_temp)

            #record = record + [class_names[labels.item()]] + [class_names[predicted.item()]]

            # print(record)
            # print(df_temp.head())
            df = df.append(df_temp)
            # break

    print(df.head())
    print("length of DF:", len(df))
    prob_file_name = "%s/%s_inference.csv" % (opt.out_dir, py_file_name)
    df.to_csv(prob_file_name, index=False)

# ======================================


if __name__ == '__main__':
    print("Started data preparation")
    data_loaders = prepare_data()
    print(vars(opt))
    print("Data is ready")

    # Train or retrain or inference
    if opt.action == "train":
        print("Training process is strted..!")
        run_train()
       # pass
    elif opt.action == "retrain":
        print("Retrainning process is strted..!")
        run_train(retrain=True)
       # pass
    elif opt.action == "test":
        print("Inference process is strted..!")
        test_model()
    elif opt.action == "check":
        check_model_graph()
        print("Check pass")
    elif opt.action == "prepare":
        prepare_prediction_file()
        print("Probability file prepared..!")
    elif opt.action == "inference":
        inference()
        print("Inference completed")

    # Finish tensorboard writer
    writer.close()
