import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot

import conv_denoiser
import measure


def plot_samples(noise_vals, noisy_input_test, denoised_images, pure_test, nu_samples=4, img_height=64, img_width=64):
    noise_prop, noise_std, noise_mean = noise_vals
    plt.rcParams.update(plt.rcParamsDefault)
    fig, axes = plt.subplots(nu_samples, 5)  # nu_samples rows and 5 columns
    fig.set_size_inches(16, 5 * nu_samples)  # Set window size

    # Set axes titles
    axes[0][0].set_title('Original image')
    axes[0][1].set_title('Noisy image')
    axes[0][2].set_title('Autoencoder denoised image')
    axes[0][3].set_title('BM3D denoised image')
    axes[0][4].set_title('NL Means denoised image')
    # Plot denoised images
    noisy_images = []
    pure_images = []
    bm3d_images = []
    nl_images = []
    for i in range(0, nu_samples):
        # Get the sample and the reconstruction
        noisy_image = noisy_input_test[i]
        pure_image = pure_test[i]
        denoised_image = denoised_images[i]
        bm3d_denoised = conv_denoiser.bm3d_denoise(
            noisy_input_test[i].reshape(img_height, img_width, 3))
        nl_denoised = conv_denoiser.nlm_denoise(
            noisy_input_test[i].reshape(img_height, img_width, 3))
        noisy_images.append(noisy_image)
        pure_images.append(pure_image)
        bm3d_images.append(bm3d_denoised)
        nl_images.append(nl_denoised)
        nl_denoised = np.squeeze(noisy_image)
        nl_denoised = np.squeeze(denoised_image)
        nl_denoised = np.squeeze(nl_denoised)
        bm3d_denoised = np.squeeze(bm3d_denoised)
        # Plot sample and reconstruciton
        axes[i][0].imshow(np.squeeze(pure_image))
        axes[i][0].set_ylabel("SSIM: {:.5f}".format(
            measure.get_image_ssim(pure_image, pure_image)))
        axes[i][1].imshow(noisy_image)
        axes[i][1].set_ylabel("SSIM: {:.5f}".format(
            measure.get_image_ssim(pure_image, noisy_image)))
        axes[i][2].imshow(denoised_image)
        axes[i][2].set_ylabel("SSIM: {:.5f}".format(
            measure.get_image_ssim(pure_image, denoised_image)))
        axes[i][3].imshow(bm3d_denoised)
        axes[i][3].set_ylabel("SSIM: {:.5f}".format(
            measure.get_image_ssim(pure_image, bm3d_denoised)))
        axes[i][4].imshow(nl_denoised)
        axes[i][4].set_ylabel("SSIM: {:.5f}".format(
            measure.get_image_ssim(pure_image, nl_denoised)))

    # Measure SSIM values for sampled images
    n1 = measure.get_set_ssim(np.array(pure_images), np.array(
        noisy_images), img_height, img_width)
    n2 = measure.get_set_ssim(np.array(pure_images),
                              denoised_images, img_height, img_width)
    n3 = measure.get_set_ssim(
        np.array(pure_images) * 255.0, np.array(bm3d_images), img_height, img_width)
    n4 = measure.get_set_ssim(
        np.array(pure_images) * 255.0, np.array(nl_images), img_height, img_width)

    n5 = measure.get_set_psnr(np.array(pure_images), np.array(
        noisy_images), img_height, img_width)
    n6 = measure.get_set_psnr(np.array(pure_images),
                              denoised_images, img_height, img_width)
    n7 = measure.get_set_psnr(
        np.array(pure_images) * 255.0, np.array(bm3d_images), img_height, img_width)
    n8 = measure.get_set_psnr(
        np.array(pure_images) * 255.0, np.array(nl_images), img_height, img_width)
    print("Noisy SSIM: {0}".format(n1))
    print("Denoised SSIM: {0}".format(n2))
    print("BM3D SSIM: {0}".format(n3))
    print("NL Means SSIM: {0}".format(n4))

    print("Noisy PSNR: {0}".format(n5))
    print("Denoised PSNR: {0}".format(n6))
    print("BM3D PSNR: {0}".format(n7))
    print("NL Means PSNR: {0}".format(n8))
    fig.suptitle(
        "Medical Image Denoiser\nNoise Proportion: {0} - Mean: {1} - Standard Deviation: {2}\nSSIM Results-> Noisy: {3} - "
        "Denoised: {4} - BM3D: {5} - NL Means: {6} \nPSNR Results-> Noisy: {7} - Denoised: {8} - BM3D: {9} - NL Means: {10}".format(
            noise_prop, noise_mean, noise_std, n1, n2, n3, n4, n5, n6, n7, n8),
        fontsize=14,
        fontweight='bold')
    filesave = "results/output" + str(noise_prop) + "_" + str(noise_mean) + "_" + str(
        noise_std) + str(img_width) + "_" + str(img_height) + ".png"
    plt.savefig(filesave)
    # plt.show()


def save_samples(noise_vals, noisy_input_test, denoised_images, pure_test, img_height=64, img_width=64):
    noise_prop, noise_std, noise_mean = noise_vals
    plt.style.use('classic')
    fig, axes = plt.subplots(1, 5)
    fig.set_size_inches(20, 5)
    axes[0].set_title('Original image')
    axes[1].set_title('Noisy image')
    axes[2].set_title('Autoencoder denoised image')
    axes[3].set_title('BM3D denoised image')
    axes[4].set_title('NL Means denoised image')
    pure_images = []
    noisy_images = []
    bm3d_images = []
    nl_images = []
    for i in range(0, len(noisy_input_test)):
        # Get the sample and the reconstruction
        noisy_image = noisy_input_test[i]
        pure_image = pure_test[i]
        denoised_image = denoised_images[i]
        bm3d_denoised = conv_denoiser.bm3d_denoise(
            noisy_input_test[i].reshape(img_height, img_width))
        nl_denoised = conv_denoiser.nlm_denoise(
            noisy_input_test[i].reshape(img_height, img_width))
        noisy_images.append(noisy_image)
        pure_images.append(pure_image)
        bm3d_images.append(bm3d_denoised)
        nl_images.append(nl_denoised)
        # Plot sample and reconstruciton
        axes[0].imshow(pure_image, pyplot.cm.gray)
        axes[0].set_xlabel("SSIM: {:.5f}".format(
            measure.get_image_ssim(pure_image, pure_image)))
        axes[1].imshow(noisy_image, pyplot.cm.gray)
        axes[1].set_xlabel("SSIM: {:.5f}".format(
            measure.get_image_ssim(pure_image, noisy_image)))
        axes[2].imshow(denoised_image, pyplot.cm.gray)
        axes[2].set_xlabel("SSIM: {:.5f}".format(
            measure.get_image_ssim(pure_image, denoised_image)))
        axes[3].imshow(bm3d_denoised, pyplot.cm.gray)
        axes[3].set_xlabel("SSIM: {:.5f}".format(
            measure.get_image_ssim(pure_image * 255.0, bm3d_denoised)))
        axes[4].imshow(nl_denoised, pyplot.cm.gray)
        axes[4].set_xlabel("SSIM: {:.5f}".format(
            measure.get_image_ssim(pure_image * 255.0, nl_denoised)))

        fig.suptitle(
            "Medical Image Denoiser\nNoise Proportion: {0} - Mean: {1} - Standard deviation: {2}".format(noise_prop,
                                                                                                         noise_mean,
                                                                                                         noise_std),
            fontsize=14, fontweight='bold')

        plt.savefig(
            "results/img({1},{2},{3}) {0}.png".format(i, noise_prop, noise_mean, noise_std))

    n1 = measure.get_set_ssim(np.array(pure_images), np.array(
        noisy_images), img_height, img_width)
    n2 = measure.get_set_ssim(np.array(pure_images),
                              denoised_images, img_height, img_width)
    n3 = measure.get_set_ssim(
        np.array(pure_images) * 255.0, np.array(bm3d_images), img_height, img_width)
    n4 = measure.get_set_ssim(
        np.array(pure_images) * 255.0, np.array(nl_images), img_height, img_width)

    f = open(
        "results/SSIM({0},{1},{2}) Results.txt".format(noise_prop, noise_mean, noise_std), "w")
    f.write("Noise Proportion: {0} - Mean: {1} - Standard Deviation: {2}\n".format(
        noise_prop, noise_mean, noise_std))
    f.write("Noisy SSIM:" + str(n1) + "\n")
    f.write("Denoised SSIM:" + str(n2) + "\n")
    f.write("BM3D SSIM:" + str(n3) + "\n")
    f.write("NL Means SSIM:" + str(n4) + "\n")
    f.close()

    print("Noisy SSIM: {0}".format(n1))
    print("Denoised SSIM: {0}".format(n2))
    print("BM3D SSIM: {0}".format(n3))
    print("NL Means SSIM: {0}".format(n4))
