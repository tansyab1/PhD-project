# function to read two images and calculate the metrics
import numpy as np
import math
import torchmetrics as tm
import torch
import numpy as np
from numpy.linalg import inv
import pyPyrTools as ppt
from pyPyrTools.corrDn import corrDn
import math
from skimage import color, filters
import scipy
from os.path import dirname, join
import cv2


def vifvec(imref_batch, imdist_batch):
    M = 3
    subbands = [4, 7, 10, 13, 16, 19, 22, 25]
    sigma_nsq = 0.4

    batch_num = 1
    if imref_batch.ndim >= 3:
        batch_num = imref_batch.shape[0]

    vif = np.zeros([batch_num,])

    for a in range(batch_num):
        if batch_num > 1:
            imref = imref_batch[a, :, :]
            imdist = imdist_batch[a, :, :]
        else:
            imref = imref_batch
            imdist = imdist_batch

        # Wavelet Decomposition
        pyr = ppt.Spyr(imref, 4, 'sp5Filters', 'reflect1')
        org = pyr.pyr[::-1]  # reverse list

        pyr = ppt.Spyr(imdist, 4, 'sp5Filters', 'reflect1')
        dist = pyr.pyr[::-1]

        # Calculate parameters of the distortion channel
        g_all, vv_all = vif_sub_est_M(org, dist, subbands, M)

        # calculate the parameters of reference
        ssarr, larr, cuarr = refparams_vecgsm(org, subbands, M)

        num = np.zeros([1, len(subbands)])
        den = np.zeros([1, len(subbands)])

        for i in range(len(subbands)):
            sub = subbands[i]
            g = g_all[i]
            vv = vv_all[i]
            ss = ssarr[i]
            lam = larr[i]
            #cu = cuarr[i]

            #neigvals = len(lam)
            lev = math.ceil((sub - 1)/6)
            winsize = 2**lev + 1
            offset = (winsize - 1)/2
            offset = math.ceil(offset/M)

            g = g[offset:g.shape[0]-offset, offset:g.shape[1]-offset]
            vv = vv[offset:vv.shape[0]-offset, offset:vv.shape[1]-offset]
            ss = ss[offset:ss.shape[0]-offset, offset:ss.shape[1]-offset]

            temp1, temp2 = 0, 0
            rt = []
            for j in range(len(lam)):
                # distorted image information
                temp1 += np.sum(np.log2(1 + np.divide(np.multiply(
                    np.multiply(g, g), ss) * lam[j], vv + sigma_nsq)))
                # reference image information
                temp2 += np.sum(np.log2(1 + np.divide(ss * lam[j], sigma_nsq)))
                rt.append(
                    np.sum(np.log(1 + np.divide(ss * lam[j], sigma_nsq))))

            num[0, i] = temp1
            den[0, i] = temp2

        vif[a] = np.sum(num)/np.sum(den)
    print(vif)
    return vif


def vif_sub_est_M(org, dist, subbands, M):
    tol = 1e-15  # tolerance for zero variance
    g_all = []
    vv_all = []

    for i in range(len(subbands)):
        sub = subbands[i]
        y = org[sub-1]
        yn = dist[sub-1]

        # size of window used in distortion channel estimation
        lev = math.ceil((sub - 1)/6)
        winsize = 2**lev + 1
        win = np.ones([winsize, winsize])

        # force subband to be a multiple of M
        newsize = [math.floor(y.shape[0]/M) * M, math.floor(y.shape[1]/M) * M]
        y = y[:newsize[0], :newsize[1]]
        yn = yn[:newsize[0], :newsize[1]]

        # correlation with downsampling
        winstep = (M, M)
        winstart = (math.floor(M/2), math.floor(M/2))
        winstop = (y.shape[0] - math.ceil(M/2) + 1,
                   y.shape[1] - math.ceil(M/2) + 1)

        # mean
        mean_x = corrDn(y, win/np.sum(win), 'reflect1',
                        winstep, winstart, winstop)
        mean_y = corrDn(yn, win/np.sum(win), 'reflect1',
                        winstep, winstart, winstop)

        # covariance
        cov_xy = corrDn(np.multiply(y, yn), win, 'reflect1', winstep, winstart, winstop) - \
            np.sum(win) * np.multiply(mean_x, mean_y)

        # variance
        ss_x = corrDn(np.multiply(y, y), win, 'reflect1', winstep,
                      winstart, winstop) - np.sum(win) * np.multiply(mean_x, mean_x)
        ss_y = corrDn(np.multiply(yn, yn), win, 'reflect1', winstep,
                      winstart, winstop) - np.sum(win) * np.multiply(mean_y, mean_y)

        ss_x[np.where(ss_x < 0)] = 0
        ss_y[np.where(ss_y < 0)] = 0

        # Regression
        g = np.divide(cov_xy, (ss_x + tol))

        vv = (ss_y - np.multiply(g, cov_xy))/(np.sum(win))

        g[np.where(ss_x < tol)] = 0
        vv[np.where(ss_x < tol)] = ss_y[np.where(ss_x < tol)]
        ss_x[np.where(ss_x < tol)] = 0

        g[np.where(ss_y < tol)] = 0
        vv[np.where(ss_y < tol)] = 0

        g[np.where(g < 0)] = 0
        vv[np.where(g < 0)] = ss_y[np.where(g < 0)]

        vv[np.where(vv <= tol)] = tol

        g_all.append(g)
        vv_all.append(vv)

    return g_all, vv_all


def refparams_vecgsm(org, subbands, M):
    # This function caluclates the parameters of the reference image
    #l_arr = np.zeros([subbands[-1],M**2])
    l_arr, ssarr, cu_arr = [], [], []
    for i in range(len(subbands)):
        sub = subbands[i]
        y = org[sub-1]

        sizey = (math.floor(y.shape[0]/M)*M, math.floor(y.shape[1]/M)*M)
        y = y[:sizey[0], :sizey[1]]

        # Collect MxM blocks, rearrange into M^2 dimensional vector
        temp = []
        for j in range(M):
            for k in range(M):
                temp.append(
                    y[k:y.shape[0]-M+k+1, j:y.shape[1]-M+j+1].T.reshape(-1))

        temp = np.asarray(temp)
        mcu = np.mean(temp, axis=1).reshape(temp.shape[0], 1)
        mean_sub = temp - np.repeat(mcu, temp.shape[1], axis=1)
        cu = mean_sub @ mean_sub.T / temp.shape[1]
        # Calculate S field, non-overlapping blocks
        temp = []
        for j in range(M):
            for k in range(M):
                temp.append(y[k::M, j::M].T.reshape(-1))

        temp = np.asarray(temp)
        ss = inv(cu) @ temp
        ss = np.sum(np.multiply(ss, temp), axis=0)/(M**2)
        ss = ss.reshape(int(sizey[1]/M), int(sizey[0]/M)).T

        d, _ = np.linalg.eig(cu)
        l_arr.append(d)
        ssarr.append(ss)
        cu_arr.append(cu)

    return ssarr, l_arr, cu_arr


# function to calculate PSNR
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calculate_ssim(img1, img2):
    return tm.functional.structural_similarity_index_measure(torch.from_numpy(img1), torch.from_numpy(img2), data_range=255.0, reduction='mean')


# calculate Visual information fidelity (VIF)
def calculate_vif(img1, img2):
    return vifvec(img1, img2)


# calculate BRIQUE pip install brisque
def calculate_briq(img1):
    from brisque import BRISQUE
    return BRISQUE().score(img1)


def nmetrics(a):
    rgb = a
    lab = color.rgb2lab(a)
    gray = color.rgb2gray(a)
    # UCIQE
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    l = lab[:, :, 0]

    # 1st term
    chroma = (lab[:, :, 1]**2 + lab[:, :, 2]**2)**0.5
    uc = np.mean(chroma)
    sc = (np.mean((chroma - uc)**2))**0.5

    # 2nd term
    top = np.int(np.round(0.01*l.shape[0]*l.shape[1]))
    sl = np.sort(l, axis=None)
    isl = sl[::-1]
    conl = np.mean(isl[:top])-np.mean(sl[:top])

    # 3rd term
    satur = []
    chroma1 = chroma.flatten()
    l1 = l.flatten()
    for i in range(len(l1)):
        if chroma1[i] == 0:
            satur.append(0)
        elif l1[i] == 0:
            satur.append(0)
        else:
            satur.append(chroma1[i] / l1[i])

    us = np.mean(satur)

    uciqe = c1 * sc + c2 * conl + c3 * us

    # UIQM
    p1 = 0.0282
    p2 = 0.2953
    p3 = 3.5753

    # 1st term UICM
    rg = rgb[:, :, 0] - rgb[:, :, 1]
    yb = (rgb[:, :, 0] + rgb[:, :, 1]) / 2 - rgb[:, :, 2]
    rgl = np.sort(rg, axis=None)
    ybl = np.sort(yb, axis=None)
    al1 = 0.1
    al2 = 0.1
    T1 = np.int(al1 * len(rgl))
    T2 = np.int(al2 * len(rgl))
    rgl_tr = rgl[T1:-T2]
    ybl_tr = ybl[T1:-T2]

    urg = np.mean(rgl_tr)
    s2rg = np.mean((rgl_tr - urg) ** 2)
    uyb = np.mean(ybl_tr)
    s2yb = np.mean((ybl_tr - uyb) ** 2)

    uicm = -0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

    # 2nd term UISM (k1k2=8x8)
    Rsobel = rgb[:, :, 0] * filters.sobel(rgb[:, :, 0])
    Gsobel = rgb[:, :, 1] * filters.sobel(rgb[:, :, 1])
    Bsobel = rgb[:, :, 2] * filters.sobel(rgb[:, :, 2])

    Rsobel = np.round(Rsobel).astype(np.uint8)
    Gsobel = np.round(Gsobel).astype(np.uint8)
    Bsobel = np.round(Bsobel).astype(np.uint8)

    Reme = eme(Rsobel)
    Geme = eme(Gsobel)
    Beme = eme(Bsobel)

    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    # 3rd term UIConM
    uiconm = logamee(gray)

    uiqm = p1 * uicm + p2 * uism + p3 * uiconm
    return uiqm, uciqe


def eme(ch, blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)

    eme = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]

            block = ch[xlb:xrb, ylb:yrb]

            blockmin = np.float(np.min(block))
            blockmax = np.float(np.max(block))

            # # old version
            # if blockmin == 0.0: eme += 0
            # elif blockmax == 0.0: eme += 0
            # else: eme += w * math.log(blockmax / blockmin)

            # new version
            if blockmin == 0:
                blockmin += 1
            if blockmax == 0:
                blockmax += 1
            eme += w * math.log(blockmax / blockmin)
    return eme


def plipsum(i, j, gamma=1026):
    return i + j - i * j / gamma


def plipsub(i, j, k=1026):
    return k * (i - j) / (k - j)


def plipmult(c, j, gamma=1026):
    return gamma - gamma * (1 - j / gamma)**c


def logamee(ch, blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)

    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]

            block = ch[xlb:xrb, ylb:yrb]
            blockmin = np.float(np.min(block))
            blockmax = np.float(np.max(block))

            top = plipsub(blockmax, blockmin)
            bottom = plipsum(blockmax, blockmin)

            m = top/bottom
            if m == 0.:
                s += 0
            else:
                s += (m) * np.log(m)

    return plipmult(w, s)


# calculate UIQM and UCIQE for uneven illumination
def calculate_uiqm_uciqe(img):
    uiqm, uciqe = nmetrics(img)
    return uiqm, uciqe


# calculate NIQE metrics

gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0/gamma_range)
a *= a
b = scipy.special.gamma(1.0/gamma_range)
c = scipy.special.gamma(3.0/gamma_range)
prec_gammas = a/(b*c)


def aggd_features(imdata):
    # flatten imdata
    imdata.shape = (len(imdata.flat),)
    imdata2 = imdata*imdata
    left_data = imdata2[imdata < 0]
    right_data = imdata2[imdata >= 0]
    left_mean_sqrt = 0
    right_mean_sqrt = 0
    if len(left_data) > 0:
        left_mean_sqrt = np.sqrt(np.average(left_data))
    if len(right_data) > 0:
        right_mean_sqrt = np.sqrt(np.average(right_data))

    if right_mean_sqrt != 0:
        gamma_hat = left_mean_sqrt/right_mean_sqrt
    else:
        gamma_hat = np.inf
    # solve r-hat norm

    imdata2_mean = np.mean(imdata2)
    if imdata2_mean != 0:
        r_hat = (np.average(np.abs(imdata))**2) / (np.average(imdata2))
    else:
        r_hat = np.inf
    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1) *
                         (gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    # solve alpha by guessing values that minimize ro
    pos = np.argmin((prec_gammas - rhat_norm)**2)
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0/alpha)
    gam2 = scipy.special.gamma(2.0/alpha)
    gam3 = scipy.special.gamma(3.0/alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    # mean parameter
    N = (br - bl)*(gam2 / gam1)  # *aggdratio
    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)


def ggd_features(imdata):
    nr_gam = 1/prec_gammas
    sigma_sq = np.var(imdata)
    E = np.mean(np.abs(imdata))
    rho = sigma_sq/E**2
    pos = np.argmin(np.abs(nr_gam - rho))
    return gamma_range[pos], sigma_sq


def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return (H_img, V_img, D1_img, D2_img)


def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights


def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0/6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1,
                              mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image**2, avg_window, 0,
                              var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window,
                              1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image**2))
    return (image - mu_image)/(var_image + C), var_image, mu_image


def _niqe_extract_subband_feats(mscncoefs):
    # alpha_m,  = extract_ggd_features(mscncoefs)
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([alpha_m, (bl+br)/2.0,
                     alpha1, N1, bl1, br1,  # (V)
                     alpha2, N2, bl2, br2,  # (H)
                     alpha3, N3, bl3, bl3,  # (D1)
                     alpha4, N4, bl4, bl4,  # (D2)
                     ])


def get_patches_train_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 1, stride)


def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)


def extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = np.int(patch_size)
    patches = []
    for j in range(0, h-patch_size+1, patch_size):
        for i in range(0, w-patch_size+1, patch_size):
            patch = img[j:j+patch_size, i:i+patch_size]
            patches.append(patch)

    patches = np.array(patches)

    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)

    return patch_features


def _get_patches_generic(img, patch_size, is_train, stride):
    h, w = np.shape(img)
    if h < patch_size or w < patch_size:
        print("Input image is too small")
        exit(0)

    # ensure that the patch divides evenly into img
    hoffset = (h % patch_size)
    woffset = (w % patch_size)

    if hoffset > 0:
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]

    img = img.astype(np.float32)
    img2 = scipy.misc.imresize(img, 0.5, interp='bicubic', mode='F')

    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(np.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2)
    mscn2 = mscn2.astype(np.float32)

    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size/2)

    feats = np.hstack((feats_lvl1, feats_lvl2))  # feats_lvl3))

    return feats


def niqe(inputImgData):

    patch_size = 96
    module_path = dirname(__file__)

    # TODO: memoize
    params = scipy.io.loadmat(
        join(module_path, 'data', 'niqe_image_params.mat'))
    pop_mu = np.ravel(params["pop_mu"])
    pop_cov = params["pop_cov"]

    M, N = inputImgData.shape

    # assert C == 1, "niqe called with videos containing %d channels. Please supply only the luminance channel" % (C,)
    assert M > (patch_size*2+1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"
    assert N > (patch_size*2+1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"

    feats = get_patches_test_features(inputImgData, patch_size)
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)

    X = sample_mu - pop_mu
    covmat = ((pop_cov+sample_cov)/2.0)
    pinvmat = scipy.linalg.pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))

    return niqe_score


# calculate the discrete entropy of a 2D image 
def entropy(img):
    # calculate the histogram
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    # normalize the histogram
    hist_norm = hist.ravel()/hist.max()
    # calculate the discrete entropy
    eps = 1e-10
    H = -np.sum(hist_norm*np.log2(hist_norm+eps))
    return H


# function to call all the above functions

def calculatemetrics(img_ref, img):
    psnr = calculate_psnr(img_ref, img)
    ssim = calculate_ssim(img_ref, img)
    vif = vifvec(img_ref, img)
    uiqm, uciqe = calculate_uiqm_uciqe(img)
    niqe_idx = niqe(img)
    entropy_idx = entropy(img)
    
