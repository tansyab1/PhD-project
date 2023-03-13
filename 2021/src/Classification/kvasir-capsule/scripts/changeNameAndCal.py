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
    path = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/forNoise/'
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

    # dict_briq_ui = {'LIME': 0, 'RetinexNet': 0, 'EnlightenGAN': 0,
    #                 'MIRNet': 0, 'Uformer': 0, 'TCFA': 0}
    # dict_niqe_ui = {'LIME': 0, 'RetinexNet': 0, 'EnlightenGAN': 0,
    #                 'MIRNet': 0, 'Uformer': 0, 'TCFA': 0}
    # dict_entropy_ui = {'LIME': 0, 'RetinexNet': 0, 'EnlightenGAN': 0,
    #                    'MIRNet': 0, 'Uformer': 0, 'TCFA': 0}

    # methods_ui = ['LIME', 'RetinexNet',
    #               'EnlightenGAN', 'MIRNet', 'Uformer', 'TCFA']

    brisque_method = {'BM3D': [], 'CycleISP': [], 'DANet': [], 'MPRNet': [], 'RIDNet': [], 'VDNet': [], 'Uformer': [], 'TCFA': []}
    niqe_method = {'BM3D': [], 'CycleISP': [], 'DANet': [], 'MPRNet': [], 'RIDNet': [], 'VDNet': [], 'Uformer': [], 'TCFA': []}
    entropy_method = {'BM3D': [], 'CycleISP': [], 'DANet': [], 'MPRNet': [], 'RIDNet': [], 'VDNet': [], 'Uformer': [], 'TCFA': []}
    mean_entropy = {}
    std_entropy = {}
    mean_brisque = {}
    std_brisque = {}
    mean_niqe = {}
    std_niqe = {}

    pathimgname = path + 'CycleISP/'

    for imgname in tqdm(glob(pathimgname + '/*.png')):
        fileBaseName = os.path.basename(imgname)[0:-4]
        for method in methods_noise:
            # find the file with the same name
            for file in glob(path + method + '/*'):
                if fileBaseName in file:
                    # if np.shape(cv2.imread(file)) != np.shape(cv2.imread(imgname)):
                    #     print(np.shape(cv2.imread(file)), np.shape(cv2.imread(imgname)))
                    #     continue
                    img = cv2.imread(file)
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # calculate the value of the metric
                    # entropy_method[method].append(entropy(cv2.imread(file)))
                    # mean_psnr[method] = np.mean(entropy_method[method])
                    entropy_method[method].append(entropy(img_gray))
                    brisque_method[method].append(calculate_briq(img))
                    niqe_method[method].append(calculate_niqe(img,0))
                    # brisque_method[method] = calculate_briq(cv2.imread(file))
                    # print(method, briq)
        for method in methods_noise:
            # if entropy_method[method] == np.max(list(entropy_method.values())):
            #     dict_entropy_ui[method] += 1
            mean_entropy[method] = np.mean(entropy_method[method])
            std_entropy[method] = np.std(entropy_method[method])
            mean_brisque[method] = np.mean(brisque_method[method])
            std_brisque[method] = np.std(brisque_method[method])
            mean_niqe[method] = np.mean(niqe_method[method])
            std_niqe[method] = np.std(niqe_method[method])
            

            # if niqe_method[method] == np.min(list(niqe_method.values())):
            #     dict_niqe_ui[method] += 1

            # if brisque_method[method] == np.max(list(brisque_method.values())):
            #     dict_briq_ui[method] += 1

    # print(dict_briq_ui)
    # print(dict_niqe_ui)
    # print(dict_entropy_ui)

    # # save the result to a file
    f = open(path + 'brisque_std.txt', 'w')
    f.write(str(std_entropy))
    f.write(str(mean_entropy))
    
    f = open(path + 'niqe_std.txt', 'w')
    f.write(str(std_niqe))
    f.write(str(mean_niqe))
    
    f = open(path + 'entropy_std.txt', 'w')
    f.write(str(std_brisque))
    f.write(str(mean_brisque))

    # f = open(path + 'niqe_ui.txt', 'w')
    # f.write(str(dict_niqe_ui))

    # f = open(path + 'brisque_ui.txt', 'w')
    # f.write(str(dict_briq_ui))
