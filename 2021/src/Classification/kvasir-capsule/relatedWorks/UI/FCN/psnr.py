import torch
from torchmetrics.functional import structural_similarity_index_measure as ssim_func
import cv2

def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def batch_PSNR(img1, img2, data_range=None):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR)

def batch_SSIM(img1, img2, data_range=None):
    SSIM = []
    # print(img1.shape, img2.shape)
    # ssim_func = StructuralSimilarityIndexMeasure(data_range=255)
    for im1, im2 in zip(img1, img2):
        # unsqueeze(0) is for batch size
        im1 = im1.unsqueeze(0)
        im2 = im2.unsqueeze(0)
        
        ssim = ssim_func(im1, im2)
        SSIM.append(ssim)
        # print(ssim)
    return sum(SSIM)/len(SSIM)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))