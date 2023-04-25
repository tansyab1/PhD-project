from config import Config
import argparse
from model import FCN
import numpy as np
import time
import random
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import os
from losses import CharbonnierLoss, SSIMLoss
from dataloader import get_training_data, get_validation_data
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from psnr import batch_PSNR, batch_SSIM

argparser = argparse.ArgumentParser(description='PyTorch FCN Training')

opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

torch.backends.cudnn.benchmark = True


######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1

mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

os.makedirs(result_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir = opt.TRAINING.VAL_DIR
save_images = opt.TRAINING.SAVE_IMAGES

# load model and put on GPU
model_restoration = FCN()
model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")


new_lr = opt.OPTIM.LR_INITIAL

optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(
    0.9, 0.999), eps=1e-8, weight_decay=1e-8)

warmup = True

if warmup:
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

print('===> Start Epoch {} End Epoch {}'.format(
    start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

train_dataset = get_training_data(train_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE,
                          shuffle=True, num_workers=16, drop_last=False)

val_dataset = get_validation_data(val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=16,
                        shuffle=False, num_workers=8, drop_last=False)

criterionl1 = CharbonnierLoss().cuda()
criterionssim = SSIMLoss().cuda()

best_psnr = 0
best_epoch = 0
best_iter = 0

eval_now = len(train_loader)//4 - 1
print(f"\nEvaluation after every {eval_now} Iterations !!!\n")

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0

    for i, data in enumerate(tqdm(train_loader), 0):
        target = data[0].cuda()
        input_ = data[1].cuda()

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        restored = model_restoration(input_)
        restored = torch.clamp(restored, 0, 1)

        loss_l1 = criterionl1(restored, target)
        loss_ssim = criterionssim(restored, target)

        loss = loss_l1 + loss_ssim

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if i % eval_now == 0 and i > 0:
            model_restoration.eval()
            with torch.no_grad():
                psnr_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    filenames = data_val[2]

                    restored = model_restoration(input_)
                    restored = torch.clamp(restored, 0, 1)
                    psnr_val_rgb.append(batch_PSNR(restored, target, 1.))

                psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)

                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer': optimizer.state_dict()
                                }, os.path.join(model_dir, "model_best.pth"))

                print("[Ep %d it %d\t PSNR FCN: %.4f\t] ----  [best_Ep_FCN %d best_it_FCN %d Best_PSNR_FCN %.4f] " %
                      (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))

            model_restoration.train()

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(
        epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))
    
    if epoch % 10 == 0:
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
