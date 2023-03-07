import os
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
from niqe import calculate_niqe
from calculate_metrics import calculate_psnr, calculate_ssim

# define a function to remove the file with the suffix


def remove_file(file_path, suffix):
    for file in glob(file_path + '/*' + suffix):
        os.remove(file)


# function to change the name of the file with the suffix
def change_name(file_path, suffix):
    for file in tqdm(glob(file_path + '/*' + suffix)):
        os.rename(file, file.replace(suffix, '.jpg'))


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


if __name__ == '__main__':

    # define the path of the folder
    path = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/forUI/'
    # define dictionary to store the name of the method and the value of the metric
    dict_briq_noise = {'BM3D': 0, 'CycleISP': 0, 'DANet': 0,
                       'MPRNet': 0, 'RIDNet': 0, 'VDNet': 0, 'Uformer': 0, 'TCFA': 0}
    dict_niqe_noise = {'BM3D': 0, 'CycleISP': 0, 'DANet': 0,
                       'MPRNet': 0, 'RIDNet': 0, 'VDNet': 0, 'Uformer': 0, 'TCFA': 0}
    dict_entropy_noise = {'BM3D': 0, 'CycleISP': 0, 'DANet': 0,
                          'MPRNet': 0, 'RIDNet': 0, 'VDNet': 0, 'Uformer': 0, 'TCFA': 0}
    methods_noise = ['BM3D', 'CycleISP', 'DANet', 'MPRNet',
                     'RIDNet', 'VDNet', 'Uformer', 'TCFA']

    dict_briq_blur = {'DBGAN': 0, 'DeblurGANv2': 0,
                      'DMPHN': 0, 'MIRNet': 0, 'Uformer': 0, 'TCFA': 0}
    dict_niqe_blur = {'DBGAN': 0, 'DeblurGANv2': 0,
                      'DMPHN': 0, 'MIRNet': 0, 'Uformer': 0, 'TCFA': 0}
    dict_entropy_blur = {'DBGAN': 0, 'DeblurGANv2': 0,
                         'DMPHN': 0, 'MIRNet': 0, 'Uformer': 0, 'TCFA': 0}
    methods_blur = ['DBGAN', 'DeblurGANv2',
                    'DMPHN', 'MIRNet', 'Uformer', 'TCFA']

    dict_briq_ui = {'LIME': 0, 'RetinexNet': 0, 'EnlightenGAN': 0,
                    'MIRNet': 0, 'Uformer': 0, 'TCFA': 0}
    dict_niqe_ui = {'LIME': 0, 'RetinexNet': 0, 'EnlightenGAN': 0,
                    'MIRNet': 0, 'Uformer': 0, 'TCFA': 0}
    dict_entropy_ui = {'LIME': 0, 'RetinexNet': 0, 'EnlightenGAN': 0,
                       'MIRNet': 0, 'Uformer': 0, 'TCFA': 0}

    methods_ui = ['LIME', 'RetinexNet',
                  'EnlightenGAN', 'MIRNet', 'Uformer', 'TCFA']

    brisque_method = {}
    niqe_method = {}
    entropy_method = {'LIME': [], 'RetinexNet': [], 'EnlightenGAN': [],
                       'MIRNet': [], 'Uformer': [], 'TCFA': []}
    mean_psnr = {}
    std_psnr = {}

    pathimgname = path + 'EnlightenGAN/'

    for imgname in tqdm(glob(pathimgname + '/*.png')):
        fileBaseName = os.path.basename(imgname)[0:-4]
        for method in methods_ui:
            # find the file with the same name
            for file in glob(path + method + '/*'):
                if fileBaseName in file:
                    if np.shape(cv2.imread(file)) != np.shape(cv2.imread(imgname)):
                        continue
                    # calculate the value of the metric
                    entropy_method[method].append(calculate_psnr(cv2.imread(file), cv2.imread(imgname)))
                    # print(method, entropy_method[method])
                    # mean_psnr[method] = np.mean(entropy_method[method])
                    # niqe_method[method] = calculate_niqe(cv2.imread(file), 0)
                    # brisque_method[method] = calculate_briq(cv2.imread(file))
                    # print(method, briq)
        for method in methods_ui:
            # if entropy_method[method] == np.max(list(entropy_method.values())):
            #     dict_entropy_ui[method] += 1
            mean_psnr[method] = np.mean(entropy_method[method])
            std_psnr[method] = np.std(entropy_method[method])
            

            # if niqe_method[method] == np.min(list(niqe_method.values())):
            #     dict_niqe_ui[method] += 1

            # if brisque_method[method] == np.max(list(brisque_method.values())):
            #     dict_briq_ui[method] += 1

    # print(dict_briq_ui)
    # print(dict_niqe_ui)
    # print(dict_entropy_ui)

    # # save the result to a file
    f = open(path + 'psnr_ui.txt', 'w')
    f.write(str(mean_psnr))
    f.write(str(std_psnr))

    # f = open(path + 'niqe_ui.txt', 'w')
    # f.write(str(dict_niqe_ui))

    # f = open(path + 'brisque_ui.txt', 'w')
    # f.write(str(dict_briq_ui))
