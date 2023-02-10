import os
from glob import glob
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
# from fire import Fire
from tqdm import tqdm
from math import log10, sqrt
import matplotlib.pyplot as plt

from aug import get_normalize
from models.networks import get_generator
from skimage.metrics import structural_similarity as ssim
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class Predictor:
    def __init__(self, weights_path: str, model_name: str = ''):
        with open('config/config.yaml', encoding='utf-8') as cfg:
            config = yaml.load(cfg, Loader=yaml.FullLoader)
        model = get_generator(model_name or config['model'])
        model.load_state_dict(torch.load(weights_path)['model'])
        self.model = model.cuda()
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):
        x, _ = self.normalize_fn(x, x)
        if mask is None:
            mask = np.ones_like(x, dtype=np.float32)
        else:
            mask = np.round(mask.astype('float32') / 255)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                      }
        x = np.pad(x, **pad_params)
        mask = np.pad(mask, **pad_params)

        return map(self._array_to_batch, (x, mask)), h, w

    @staticmethod
    def _postprocess(x: torch.Tensor) -> np.ndarray:
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray], ignore_mask=True) -> np.ndarray:
        (img, mask), h, w = self._preprocess(img, mask)
        with torch.no_grad():
            inputs = [img.cuda()]
            if not ignore_mask:
                inputs += [mask]
            pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]


def process_video(pairs, predictor, output_dir):
    for video_filepath, mask in tqdm(pairs):
        video_filename = os.path.basename(video_filepath)
        output_filepath = os.path.join(
            output_dir, os.path.splitext(video_filename)[0]+'_deblur.mp4')
        video_in = cv2.VideoCapture(video_filepath)
        fps = video_in.get(cv2.CAP_PROP_FPS)
        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frame_num = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
        video_out = cv2.VideoWriter(
            output_filepath, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
        tqdm.write(
            f'process {video_filepath} to {output_filepath}, {fps}fps, resolution: {width}x{height}')
        for frame_num in tqdm(range(total_frame_num), desc=video_filename):
            res, img = video_in.read()
            if not res:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = predictor(img, mask)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            video_out.write(pred)


def main(img_pattern: str,
         mask_pattern: Optional[str] = None,
         weights_path='./models/fpn_inception.h5',
         out_dir='/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/results/DeblurGANv2/',
         side_by_side: bool = False,
         video: bool = False):
    def sorted_glob(pattern):
        return sorted(glob(pattern))

    imgs = sorted_glob(img_pattern)
    masks = sorted_glob(mask_pattern) if mask_pattern is not None else [
        None for _ in imgs]
    pairs = zip(imgs, masks)
    names = sorted([os.path.basename(x) for x in glob(img_pattern)])
    predictor = Predictor(weights_path=weights_path)

    os.makedirs(out_dir, exist_ok=True)
    if not video:
        for name, pair in tqdm(zip(names, pairs), total=len(names)):
            f_img, f_mask = pair
            img = cv2.imread(f_img)
            mask = cv2.imread(f_mask) if f_mask is not None else None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pred = predictor(img, mask)
            if side_by_side:
                pred = np.hstack((img, pred))
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, name),
                        pred)
            return pred
    else:
        process_video(pairs, predictor, out_dir)

# def getfiles():
#     filenames = os.listdir(r'.\dataset1\blur')
#     print(filenames)


def get_files(filepath=r'.\dataset1\blur'):
    list = []
    for filepath, dirnames, filenames in os.walk(filepath):
        for filename in filenames:
            list.append(os.path.join(filepath, filename))
    return list


def cal_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


if __name__ == '__main__':
    blurpath = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Blur_var/test/input"
    gt_path = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Blur_var/test/groundtruth"
    f = open("psnr-ssim.txt", "a")
    img_path = get_files(blurpath)
    gt = get_files(gt_path)
    for i in img_path:
        pre = main(i)
        # img = .imread(i)
        gt_img = cv2.imread(i.replace('input', 'groundtruth'))
        psnr = cal_psnr(gt_img, pre)
        # calculate ssim
        ssim_inx = ssim(gt_img, pre, multichannel=True)
        f.write(i.split('/')[-1] + " PSNR:" +
                str(psnr) + " SISM:" + str(ssim_inx) + "\n")
        print(i.split('/')[-1] + " PSNR:" + str(psnr) + " SISM:" + str(ssim_inx))

    # move file f to folder result and rename it to psnr-ssim.txt
    os.rename(f, os.path.join("submit", "psnr-ssim.txt"))

    # main('test_img/tt.mp4')
