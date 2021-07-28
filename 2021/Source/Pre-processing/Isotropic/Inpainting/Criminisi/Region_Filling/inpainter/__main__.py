import argparse
from skimage.io import imread, imsave

from inpainter import Inpainter


def main():
    args = parse_args()

    image = imread('../../test_cases/10.jpg')
    mask = imread('../../test_cases/mask10.jpg', as_gray=True)

    output_image = Inpainter(
        image,
        mask,
        patch_size=args.patch_size,
        plot_progress=args.plot_progress
    ).inpaint()
    imsave(args.output, output_image, quality=100)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-ps',
        '--patch-size',
        help='the size of the patches',
        type=int,
        default=9
    )
    parser.add_argument(
        '-o',
        '--output',
        help='the file path to save the output image',
        default='output.jpg'
    )
    parser.add_argument(
        '--plot-progress',
        help='plot each generated image',
        action='store_true',
        default=True
    )
    # parser.add_argument(
    #     'input_image',
    #     help='the image containing objects to be removed',
    #     default='../test_cases/10.jpg'
    # )
    # parser.add_argument(
    #     'mask',
    #     help='the mask of the region to be removed',
    #     default='../test_cases/mask10.jpg'
    # )
    return parser.parse_args()


if __name__ == '__main__':
    main()
