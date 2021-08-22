from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy as np
from bm3d import BM3D_1st_step_color, BM3D_2nd_step_color
import cv2


def nlm_denoise(noisy_image):
    noisy_image = noisy_image * 255.0
    denoised = []
    e1 = cv2.getTickCount()
    sigma_est = np.mean(estimate_sigma(noisy_image, multichannel=True, average_sigmas=True))
    denoised_image = denoise_nl_means(noisy_image, h=1*sigma_est, multichannel=True)
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()   # 计算函数执行时间
    print("The Processing time of the NL is %f s" % time)
    denoised.append(denoised_image)
    print("Image denoised using NL Means")

    return np.array(denoised)


def bm3d_denoise(noisy_image):
    noisy_image = np.float32(noisy_image * 255.0)
    denoised = []
    # cv2.setUseOptimized(True)

    # img_name_gold = "image_Lena512rgb.png"
    # img_gold = cv2.imread(img_name_gold, cv2.IMREAD_COLOR)
    # noise = numpy.random.normal(scale=sigma,
    #                             size=img_gold.shape).astype(numpy.int32)

    # img = img_gold + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = noisy_image.astype(np.uint8)

    imgYCB = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2YCrCb)
    # imgYCB_gold = cv2.cvtColor(img_gold, cv2.COLOR_BGR2YCrCb)

    # cv2.imwrite("Nosiy_sigma"+str(sigma)+"_color.png", img)

    # psnr = PSNR2(img, img_gold)
    # print("The PSNR between noisy image and ref image is %f" % psnr)

    # # 记录程序运行时间
    e1 = cv2.getTickCount()  # cv2.getTickCount 函数返回从参考点到这个函数被执行的时钟数
    Basic_img = BM3D_1st_step_color(imgYCB)

    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()   # 计算函数执行时间
    print("The Processing time of the First step is %f s" % time)

    # cv2.imwrite("Basic_sigma"+str(sigma)+"_color.png",
    #             cv2.cvtColor(Basic_img, cv2.COLOR_YCrCb2BGR))
    # psnr = PSNR2(img_gold, cv2.cvtColor(Basic_img, cv2.COLOR_YCrCb2BGR))
    # print("The PSNR compared with gold image for the First step is %f" % psnr)

    Final_img = BM3D_2nd_step_color(Basic_img, imgYCB)
    Final_img = cv2.cvtColor(Final_img, cv2.COLOR_YCrCb2RGB)
    # cv2.imwrite("Final_sigma"+str(sigma)+"_color.png",
    #             cv2.cvtColor(Final_img, cv2.COLOR_YCrCb2BGR))

    e3 = cv2.getTickCount()
    time = (e3 - e2) / cv2.getTickFrequency()
    print("The Processing time of the Second step is %f s" % time)
    # psnr = PSNR2(img_gold, cv2.cvtColor(Final_img, cv2.COLOR_YCrCb2BGR))
    # print("The PSNR compared with gold image for the Second step is %f" % psnr)

    denoised.append(Final_img)
    print("Image denoised")
    return np.array(denoised)
