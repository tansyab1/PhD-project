# [VCNet: a robust approach to blind image inpainting](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700749.pdf)
by [Yi Wang](https://shepnerd.github.io/), [Ying-Cong Chen](https://yingcong.github.io/), [Xin Tao](http://www.xtao.website/), and [Jiaya Jia](http://jiaya.me/). **The training & testing specifications will be updated**.

## Introduction
This repository gives the implementation of our method in ECCV 2020 paper, '[VCNet: a robust approach to blind image inpainting](https://arxiv.org/pdf/2003.06816.pdf)' \([supplementary file](https://drive.google.com/file/d/1LpMfWCi_W9oDyMJB45pQw4wOQm5-rpBp/view?usp=sharing)). It studies how to repair images with unknown contaminations automatically.

<img src="./media/teaser-v3.png" width="100%" alt="learned semantic layouts">

Blind inpainting is a task to automatically complete visual contents without specifying masks for missing areas in an image. Previous work assumes *known missing-region-pattern*, limiting the application scope. We instead *relax the assumption* by defining a new blind inpainting setting, making training a neural system robust against *various unknown missing region patterns*.

## Prerequisites
- Python3.5 (or higher)
- Tensorflow 1.4 (or later versions, excluding 2.x) with NVIDIA GPU or CPU
- OpenCV
- numpy
- scipy
- easydict

## For tensorflow implementations
### Pretrained models
[FFHQ-HQ_p256](https://pan.baidu.com/s/1wfaR1pzngKeNhxyrZlvJYw) trained with stroke masks. (Password: ted9)

[CelebA-HQ_p256](https://pan.baidu.com/s/1EL0qCzcASD4Qgf8bEFpf4A) trained with stroke masks. (Password: 7dzs)

## Acknowledgments
Our code benefits a lot from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) and [Generative Image Inpainting with Contextual Attention](https://github.com/JiahuiYu/generative_inpainting).

## Citation

If our method is useful for your research, please consider citing:

    @article{wang2020vcnet,
        title={VCNet: A Robust Approach to Blind Image Inpainting},
        author={Wang, Yi and Chen, Ying-Cong and Tao, Xin and Jia, Jiaya},
        journal={arXiv preprint arXiv:2003.06816},
        year={2020}
    }


### Contact
Please send email to yiwang@cse.cuhk.edu.hk.
