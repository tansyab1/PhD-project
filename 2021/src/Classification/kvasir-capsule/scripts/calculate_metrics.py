# function to read two images and calculate the metrics
import numpy as np
import math
import torchmetrics as tm
import torch
import numpy as np
from numpy.linalg import inv
from skimage.transform import resize
# import pyPyrTools as ppt
# from pyPyrTools.corrDn import corrDn
import math
from skimage import color, filters
import scipy
from os.path import dirname, join
import cv2
from glob import glob
import os
from scipy import special
from scipy import io, misc
from tqdm import tqdm
import numpy as np


def im2col(img, k, stride=1):
    # Parameters
    m, n = img.shape
    s0, s1 = img.strides
    nrows = m - k + 1
    ncols = n - k + 1
    shape = (k, k, nrows, ncols)
    arr_stride = (s0, s1, s0, s1)

    ret = np.lib.stride_tricks.as_strided(img, shape=shape, strides=arr_stride)
    return ret[:, :, ::stride, ::stride].reshape(k*k, -1)


def integral_image(x):
    M, N = x.shape
    int_x = np.zeros((M+1, N+1))
    int_x[1:, 1:] = np.cumsum(np.cumsum(x, 0), 1)
    return int_x


def moments(x, y, k, stride):
    kh = kw = k

    k_norm = k**2

    x_pad = np.pad(x, int((kh - stride)/2), mode='reflect')
    y_pad = np.pad(y, int((kw - stride)/2), mode='reflect')

    int_1_x = integral_image(x_pad)
    int_1_y = integral_image(y_pad)

    int_2_x = integral_image(x_pad*x_pad)
    int_2_y = integral_image(y_pad*y_pad)

    int_xy = integral_image(x_pad*y_pad)

    mu_x = (int_1_x[:-kh:stride, :-kw:stride] - int_1_x[:-kh:stride, kw::stride] -
            int_1_x[kh::stride, :-kw:stride] + int_1_x[kh::stride, kw::stride])/k_norm
    mu_y = (int_1_y[:-kh:stride, :-kw:stride] - int_1_y[:-kh:stride, kw::stride] -
            int_1_y[kh::stride, :-kw:stride] + int_1_y[kh::stride, kw::stride])/k_norm

    var_x = (int_2_x[:-kh:stride, :-kw:stride] - int_2_x[:-kh:stride, kw::stride] -
             int_2_x[kh::stride, :-kw:stride] + int_2_x[kh::stride, kw::stride])/k_norm - mu_x**2
    var_y = (int_2_y[:-kh:stride, :-kw:stride] - int_2_y[:-kh:stride, kw::stride] -
             int_2_y[kh::stride, :-kw:stride] + int_2_y[kh::stride, kw::stride])/k_norm - mu_y**2

    cov_xy = (int_xy[:-kh:stride, :-kw:stride] - int_xy[:-kh:stride, kw::stride] -
              int_xy[kh::stride, :-kw:stride] + int_xy[kh::stride, kw::stride])/k_norm - mu_x*mu_y

    mask_x = (var_x < 0)
    mask_y = (var_y < 0)

    var_x[mask_x] = 0
    var_y[mask_y] = 0

    cov_xy[mask_x + mask_y] = 0

    return (mu_x, mu_y, var_x, var_y, cov_xy)


def vif_gsm_model(pyr, subband_keys, M):
    tol = 1e-15
    s_all = []
    lamda_all = []

    for subband_key in subband_keys:
        y = pyr[subband_key]
        y_size = (int(y.shape[0]/M)*M, int(y.shape[1]/M)*M)
        y = y[:y_size[0], :y_size[1]]

        y_vecs = im2col(y, M, 1)
        cov = np.cov(y_vecs)
        lamda, V = np.linalg.eigh(cov)
        lamda[lamda < tol] = tol
        cov = V@np.diag(lamda)@V.T

        y_vecs = im2col(y, M, M)

        s = np.linalg.inv(cov)@y_vecs
        s = np.sum(s * y_vecs, 0)/(M*M)
        s = s.reshape((int(y_size[0]/M), int(y_size[1]/M)))

        s_all.append(s)
        lamda_all.append(lamda)

    return s_all, lamda_all


def vif_channel_est(pyr_ref, pyr_dist, subband_keys, M):
    tol = 1e-15
    g_all = []
    sigma_vsq_all = []

    for i, subband_key in enumerate(subband_keys):
        y_ref = pyr_ref[subband_key]
        y_dist = pyr_dist[subband_key]

        lev = int(np.ceil((i+1)/2))
        winsize = 2**lev + 1

        y_size = (int(y_ref.shape[0]/M)*M, int(y_ref.shape[1]/M)*M)
        y_ref = y_ref[:y_size[0], :y_size[1]]
        y_dist = y_dist[:y_size[0], :y_size[1]]

        mu_x, mu_y, var_x, var_y, cov_xy = moments(y_ref, y_dist, winsize, M)

        g = cov_xy / (var_x + tol)
        sigma_vsq = var_y - g*cov_xy

        g[var_x < tol] = 0
        sigma_vsq[var_x < tol] = var_y[var_x < tol]
        var_x[var_x < tol] = 0

        g[var_y < tol] = 0
        sigma_vsq[var_y < tol] = 0

        sigma_vsq[g < 0] = var_y[g < 0]
        g[g < 0] = 0

        sigma_vsq[sigma_vsq < tol] = tol

        g_all.append(g)
        sigma_vsq_all.append(sigma_vsq)

    return g_all, sigma_vsq_all


def vif(img_ref, img_dist, wavelet='steerable', full=False):
    assert wavelet in ['steerable', 'haar', 'db2',
                       'bio2.2'], 'Invalid choice of wavelet'
    M = 3
    sigma_nsq = 0.4

    if wavelet == 'steerable':
        from pyrtools.pyramids import SteerablePyramidSpace as SPyr
        pyr_ref = SPyr(img_ref, 4, 5, 'reflect1').pyr_coeffs
        pyr_dist = SPyr(img_dist, 4, 5, 'reflect1').pyr_coeffs
        subband_keys = []
        for key in list(pyr_ref.keys())[1:-2:3]:
            subband_keys.append(key)
    else:
        from pywt import wavedec2
        ret_ref = wavedec2(img_ref, wavelet, 'reflect', 4)
        ret_dist = wavedec2(img_dist, wavelet, 'reflect', 4)
        pyr_ref = {}
        pyr_dist = {}
        subband_keys = []
        for i in range(4):
            pyr_ref[(3-i, 0)] = ret_ref[i+1][0]
            pyr_ref[(3-i, 1)] = ret_ref[i+1][1]
            pyr_dist[(3-i, 0)] = ret_dist[i+1][0]
            pyr_dist[(3-i, 1)] = ret_dist[i+1][1]
            subband_keys.append((3-i, 0))
            subband_keys.append((3-i, 1))
        pyr_ref[4] = ret_ref[0]
        pyr_dist[4] = ret_dist[0]

    subband_keys.reverse()
    n_subbands = len(subband_keys)

    [g_all, sigma_vsq_all] = vif_channel_est(
        pyr_ref, pyr_dist, subband_keys, M)

    [s_all, lamda_all] = vif_gsm_model(pyr_ref, subband_keys, M)

    nums = np.zeros((n_subbands,))
    dens = np.zeros((n_subbands,))
    for i in range(n_subbands):
        g = g_all[i]
        sigma_vsq = sigma_vsq_all[i]
        s = s_all[i]
        lamda = lamda_all[i]

        n_eigs = len(lamda)

        lev = int(np.ceil((i+1)/2))
        winsize = 2**lev + 1
        offset = (winsize - 1)/2
        offset = int(np.ceil(offset/M))

        g = g[offset:-offset, offset:-offset]
        sigma_vsq = sigma_vsq[offset:-offset, offset:-offset]
        s = s[offset:-offset, offset:-offset]

        for j in range(n_eigs):
            nums[i] += np.mean(np.log(1 + g*g*s*lamda[j] /
                               (sigma_vsq+sigma_nsq)))
            dens[i] += np.mean(np.log(1 + s*lamda[j]/sigma_nsq))

    if not full:
        return np.mean(nums + 1e-4)/np.mean(dens + 1e-4)
    else:
        return np.mean(nums + 1e-4)/np.mean(dens + 1e-4), (nums + 1e-4), (dens + 1e-4)


def vif_spatial(img_ref, img_dist, k=11, sigma_nsq=0.1, stride=1, full=False):
    x = img_ref.astype('float32')
    y = img_dist.astype('float32')

    mu_x, mu_y, var_x, var_y, cov_xy = moments(x, y, k, stride)

    g = cov_xy / (var_x + 1e-10)
    sv_sq = var_y - g * cov_xy

    g[var_x < 1e-10] = 0
    sv_sq[var_x < 1e-10] = var_y[var_x < 1e-10]
    var_x[var_x < 1e-10] = 0

    g[var_y < 1e-10] = 0
    sv_sq[var_y < 1e-10] = 0

    sv_sq[g < 0] = var_x[g < 0]
    g[g < 0] = 0
    sv_sq[sv_sq < 1e-10] = 1e-10

    vif_val = np.sum(np.log(1 + g**2 * var_x / (sv_sq + sigma_nsq)) +
                     1e-4)/np.sum(np.log(1 + var_x / sigma_nsq) + 1e-4)
    if (full):
        # vif_map = (np.log(1 + g**2 * var_x / (sv_sq + sigma_nsq)) + 1e-4)/(np.log(1 + var_x / sigma_nsq) + 1e-4)
        # return (vif_val, vif_map)
        return (np.sum(np.log(1 + g**2 * var_x / (sv_sq + sigma_nsq)) + 1e-4), np.sum(np.log(1 + var_x / sigma_nsq) + 1e-4), vif_val)
    else:
        return vif_val


def msvif_spatial(img_ref, img_dist, k=11, sigma_nsq=0.1, stride=1, full=False):
    x = img_ref.astype('float32')
    y = img_dist.astype('float32')

    n_levels = 5
    nums = np.ones((n_levels,))
    dens = np.ones((n_levels,))
    for i in range(n_levels-1):
        if np.min(x.shape) <= k:
            break
        nums[i], dens[i], _ = vif_spatial(
            x, y, k, sigma_nsq, stride, full=True)
        x = x[:(x.shape[0]//2)*2, :(x.shape[1]//2)*2]
        y = y[:(y.shape[0]//2)*2, :(y.shape[1]//2)*2]
        x = (x[::2, ::2] + x[1::2, ::2] + x[1::2, 1::2] + x[::2, 1::2])/4
        y = (y[::2, ::2] + y[1::2, ::2] + y[1::2, 1::2] + y[::2, 1::2])/4

    if np.min(x.shape) > k:
        nums[-1], dens[-1], _ = vif_spatial(x,
                                            y, k, sigma_nsq, stride, full=True)
    msvifval = np.sum(nums) / np.sum(dens)

    if full:
        return msvifval, nums, dens
    else:
        return msvifval


# function to calculate PSNR
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calculate_ssim(img1, img2):
    img1 = torch.from_numpy(img1).unsqueeze(0).float()
    img1 = img1.permute(0, 3, 1, 2)
    img2 = torch.from_numpy(img2).unsqueeze(0).float()
    img2 = img2.permute(0, 3, 1, 2)
    # print(img1.shape, img2.shape)
    return tm.functional.structural_similarity_index_measure(img1, img2, data_range=255.0)


# calculate BRIQUE pip install brisque
def calculate_briq(img1):
    from brisque import BRISQUE
    return BRISQUE().score(img1)


# calculate the discrete entropy of a 2D image
def entropy(img):
    # calculate the histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # normalize the histogram
    hist_norm = hist.ravel()/hist.max()
    # calculate the discrete entropy
    eps = 1e-10
    H = -np.sum(hist_norm*np.log2(hist_norm+eps))
    return H


# function to call all the above functions

def calculatemetrics(img_ref, img):
    # psnr = calculate_psnr(img_ref, img)
    # ssim = calculate_ssim(img_ref, img)
    img_ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vif_inx = vif(img_ref_gray, img_gray)
    # vif = 0
    # uiqm, uciqe = calculate_uiqm_uciqe(img)
    # uiqm, uciqe = 0, 0
    # brique = calculate_briq(img)
    # niqe_idx = niqe(img)
    # entropy_idx = entropy(img)
    psnr = 0
    ssim = 0
    entropy_idx = 0
    brique = 0
    return psnr, ssim, vif_inx, entropy_idx, brique


if __name__ == "__main__":
    psnrs = 0
    ssims = 0
    vifs = 0
    briques = 0
    entropys = 0
    # open txt file to write the results
    
    in_path_CycleISP = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/CycleISP/results/png/"
    in_path_DANet = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/DANet/results/"
    in_path_DBGAN = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/DBGAN/experiments/save/results/PSNR_GoPro/GoPro/"
    in_path_DBGANv2 = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/DeblurGANv2/"
    in_path_Uformer_noise = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/Uformer/results/denoising/Noise_var/Uformer_B/"
    in_path_Uformer_blur = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/Uformer/results/deblurring/Blur_var/Uformer_B/"
    in_path_Uformer_noise_latest = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/Uformer/results/denoising/Noise_var/Uformer_B_latest/"
    in_path_Uformer_blur_latest = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/Uformer/results/deblurring/Blur_var/Uformer_B_latest/"
    in_path_Uformer_UI = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/Uformer/results/UI_var/Uformer_B/"
    in_path_Uformer_UI_latest = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/Uformer/results/UI_var/Uformer_B_latest/"
    in_path_RIDNet = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/RIDNet/TestCode/experiment/Noise/results/"
    ref_path = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Noise_var/test/groundtruth/"
    
    vifs_CycleISP = 0
    vifs_DANet = 0
    vifs_DBGAN = 0
    vifs_DBGANv2 = 0
    vifs_Uformer_noise = 0
    vifs_Uformer_blur = 0
    vifs_Uformer_noise_latest = 0
    vifs_Uformer_blur_latest = 0
    vifs_Uformer_UI = 0
    vifs_Uformer_UI_latest = 0
    vifs_RIDNet = 0
    
    f_CycleISP = open("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/CycleISP.txt", "w")
    f_DANet = open("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/DANet.txt", "w")
    f_DBGAN = open("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/DBGAN.txt", "w")
    f_DBGANv2 = open("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/DBGANv2.txt", "w")
    f_Uformer_noise = open("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/Uformer_noise.txt", "w")
    f_Uformer_blur = open("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/Uformer_blur.txt", "w")
    f_Uformer_noise_latest = open("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/Uformer_noise_latest.txt", "w")
    f_Uformer_blur_latest = open("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/Uformer_blur_latest.txt", "w")
    f_Uformer_UI = open("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/Uformer_UI.txt", "w")
    f_Uformer_UI_latest = open("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/Uformer_UI_latest.txt", "w")
    f_RIDNet = open("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/RIDNet.txt", "w")
    for name in tqdm(os.listdir(in_path_CycleISP)):
        # check if the file is an image
        if not name.endswith(('.png', '.jpg', '.jpeg', '.PNG', '_x1_SR.png')):
            continue
        img_CycleISP = cv2.imread(in_path_CycleISP + name)
        img_DANet = cv2.imread(in_path_DANet + name.replace(".png", ".jpg"))
        img_DBGAN = cv2.imread(in_path_DBGAN + name)
        img_DBGANv2 = cv2.imread(in_path_DBGANv2 + name.replace(".png", ".jpg"))
        img_Uformer_noise = cv2.imread(in_path_Uformer_noise + name.replace(".png", ".PNG"))
        img_Uformer_blur = cv2.imread(in_path_Uformer_blur + name.replace(".png", ".PNG"))
        img_Uformer_noise_latest = cv2.imread(in_path_Uformer_noise_latest + name.replace(".png", ".PNG"))
        img_Uformer_blur_latest = cv2.imread(in_path_Uformer_blur_latest + name.replace(".png", ".PNG"))
        img_Uformer_UI = cv2.imread(in_path_Uformer_UI + name.replace(".png", ".PNG"))
        img_Uformer_UI_latest = cv2.imread(in_path_Uformer_UI_latest + name.replace(".png", ".PNG"))
        img_RIDNet = cv2.imread(in_path_RIDNet + name.replace(".png", "_x1_SR.png"))
        
        img_ref = cv2.imread(ref_path + name.replace(".png", ".jpg"))
        
        psnr, ssim, vif_ref_CycleISP, entropy_idx, brique = calculatemetrics(img_ref, img_CycleISP)
        psnr, ssim, vif_ref_DANet, entropy_idx, brique = calculatemetrics(img_ref, img_DANet)
        psnr, ssim, vif_ref_DBGAN, entropy_idx, brique = calculatemetrics(img_ref, img_DBGAN)
        psnr, ssim, vif_ref_DBGANv2, entropy_idx, brique = calculatemetrics(img_ref, img_DBGANv2)
        psnr, ssim, vif_ref_Uformer_noise, entropy_idx, brique = calculatemetrics(img_ref, img_Uformer_noise)
        psnr, ssim, vif_ref_Uformer_blur, entropy_idx, brique = calculatemetrics(img_ref, img_Uformer_blur)
        psnr, ssim, vif_ref_Uformer_noise_latest, entropy_idx, brique = calculatemetrics(img_ref, img_Uformer_noise_latest)
        psnr, ssim, vif_ref_Uformer_blur_latest, entropy_idx, brique = calculatemetrics(img_ref, img_Uformer_blur_latest)
        psnr, ssim, vif_ref_Uformer_UI, entropy_idx, brique = calculatemetrics(img_ref, img_Uformer_UI)
        psnr, ssim, vif_ref_Uformer_UI_latest, entropy_idx, brique = calculatemetrics(img_ref, img_Uformer_UI_latest)
        psnr, ssim, vif_ref_RIDNet, entropy_idx, brique = calculatemetrics(img_ref, img_RIDNet)
        # print("psnr: ", psnr)
        # print("ssim: ", ssim)
        # print("vif: ", vif_ref)
        # print("brique: ", brique)
        # print("entropy: ", entropy_idx)
        # psnrs += psnr
        # ssims += ssim
        vifs_CycleISP += vif_ref_CycleISP
        vifs_DANet += vif_ref_DANet
        vifs_DBGAN += vif_ref_DBGAN
        vifs_DBGANv2 += vif_ref_DBGANv2
        vifs_Uformer_noise += vif_ref_Uformer_noise
        vifs_Uformer_blur += vif_ref_Uformer_blur
        vifs_Uformer_noise_latest += vif_ref_Uformer_noise_latest
        vifs_Uformer_blur_latest += vif_ref_Uformer_blur_latest
        vifs_Uformer_UI += vif_ref_Uformer_UI
        vifs_Uformer_UI_latest += vif_ref_Uformer_UI_latest
        vifs_RIDNet += vif_ref_RIDNet
        # briques += brique
        # entropys += entropy_idx

        f_CycleISP.write(name + " VIF: " + str(vif_ref_CycleISP) + "\n")
        f_DANet.write(name + " VIF: " + str(vif_ref_DANet) + "\n")
        f_DBGAN.write(name + " VIF: " + str(vif_ref_DBGAN) + "\n")
        f_DBGANv2.write(name + " VIF: " + str(vif_ref_DBGANv2) + "\n")
        f_Uformer_noise.write(name + " VIF: " + str(vif_ref_Uformer_noise) + "\n")
        f_Uformer_blur.write(name + " VIF: " + str(vif_ref_Uformer_blur) + "\n")
        f_Uformer_noise_latest.write(name + " VIF: " + str(vif_ref_Uformer_noise_latest) + "\n")
        f_Uformer_blur_latest.write(name + " VIF: " + str(vif_ref_Uformer_blur_latest) + "\n")
        f_Uformer_UI.write(name + " VIF: " + str(vif_ref_Uformer_UI) + "\n")
        f_Uformer_UI_latest.write(name + " VIF: " + str(vif_ref_Uformer_UI_latest) + "\n")
        f_RIDNet.write(name + " VIF: " + str(vif_ref_RIDNet) + "\n")

    print("Average VIF for CycleISP: ", vifs_CycleISP / len(os.listdir(in_path_CycleISP)))
    print("Average VIF for DANet: ", vifs_DANet / len(os.listdir(in_path_DANet)))
    print("Average VIF for DBGAN: ", vifs_DBGAN / len(os.listdir(in_path_DBGAN)))
    print("Average VIF for DBGANv2: ", vifs_DBGANv2 / len(os.listdir(in_path_DBGANv2)))
    print("Average VIF for Uformer_noise: ", vifs_Uformer_noise / len(os.listdir(in_path_Uformer_noise)))
    print("Average VIF for Uformer_blur: ", vifs_Uformer_blur / len(os.listdir(in_path_Uformer_blur)))
    print("Average VIF for Uformer_noise_latest: ", vifs_Uformer_noise_latest / len(os.listdir(in_path_Uformer_noise_latest)))
    print("Average VIF for Uformer_blur_latest: ", vifs_Uformer_blur_latest / len(os.listdir(in_path_Uformer_blur_latest)))
    print("Average VIF for Uformer_UI: ", vifs_Uformer_UI / len(os.listdir(in_path_Uformer_UI)))
    print("Average VIF for Uformer_UI_latest: ", vifs_Uformer_UI_latest / len(os.listdir(in_path_Uformer_UI_latest)))
    print("Average VIF for RIDNet: ", vifs_RIDNet / len(os.listdir(in_path_RIDNet)))
