
import cv2
import numpy as np
from skimage.util import random_noise

def create_noise(image,var=0.01,mean=0):
	noise_img = random_noise(img, mode='gaussian',mean=mean, var=var,clip=False)
	return np.array(255*noise_img, dtype = 'uint8')

if __name__ == "__main__":
	# Load the image
	img = cv2.imread("results/test.jpg")
	noise_img = create_noise(img)
	print((noise_img))
	# Display the noise image
	cv2.imshow('blur',noise_img)
	cv2.imwrite('AWGN.png',noise_img)
	cv2.waitKey(0)