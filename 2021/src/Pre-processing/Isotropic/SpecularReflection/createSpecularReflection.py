import numpy as np
import sys


def createARandomMask(image, mask_size):
    height, width = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            mask[i, j] = np.random.randint(0, mask_size)
    return mask


def createRandomSpecularReflection(image, mask_size):
    mask = createARandomMask(image, mask_size)
    height, width = image.shape
    specular_reflection_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if image[i, j] == 0:
                specular_reflection_image[i, j] = 0
            else:
                specular_reflection_image[i, j] = image[i, j] * \
                    mask[i, j] * np.random.uniform(0.5, 1.5)
    return specular_reflection_image


def main():
    image_path = sys.argv[1]
    specular_reflection_coeff = float(sys.argv[2])
    image = np.array(np.load(image_path))
    specular_reflection_image = createRandomSpecularReflection(
        image, specular_reflection_coeff)
    np.save(image_path.replace(".npy", "_specular_reflection.npy"),
            specular_reflection_image)


if __name__ == "__main__":
    main()
