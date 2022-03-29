import numpy as np
import math

# create a new mask

class createUnevenIllumination:
    def __init__(self, gradient_shape, max_intensity, transparency, mode):
        self.gradient_shape = gradient_shape
        self.max_intensity = max_intensity
        self.transparency = transparency
        self.mode = mode

    def create_oval(self, image, position, theta):
        st, ct = math.sin(theta), math.cos(theta)
        aa, bb = self.gradient_shape[0]**2, self.gradient_shape[1]**2

        weights = np.zeros(
            (np.shape(image)[0], np.shape(image)[1]), np.float64)
        for x in range(np.shape(image)[0]):
            for y in range(np.shape(image)[1]):
                weights[x, y] = ((((x-position[0]) * ct + (y-position[1]) * st) ** 2) / aa
                                 + (((x-position[0]) * st - (y-position[1]) * ct) ** 2) / bb)

        return np.clip(1.0 - weights, 0, 1) * self.max_intensity

    def create_mask(self, given_size, center, gradient_shape):
        mask = np.zeros((gradient_shape, gradient_shape), dtype=np.uint8)
        imgsize = mask.shape[:2]

        # scale of intensity
        innerColor = 0
        outerColor = self.max_intensity

        for y in range(imgsize[1]):
            for x in range(imgsize[0]):
                # Find the distance to the center
                distanceToCenter = np.sqrt(
                    (x - imgsize[0]//2) ** 2 + (y - imgsize[1]//2) ** 2)

                # Make it on a scale from 0 to 1innerColor
                distanceToCenter = distanceToCenter / \
                    (np.sqrt(1) * imgsize[0]/2)

                # Calculate intensity values
                intensity = innerColor * distanceToCenter + \
                    outerColor * (1 - distanceToCenter)
                if distanceToCenter < 1:
                    mask[y, x] = int(intensity)

        result = np.zeros((given_size[0], given_size[1]), dtype=np.uint8)

        ori_left = [x-int(gradient_shape//2) for x in center]
        ori_right = [x+int(gradient_shape//2) for x in center]

        # result image
        left_corner = [center[i]-int(gradient_shape//2) if center[i] -
                       int(gradient_shape//2) >= 0 else 0 for i in range(2)]
        right_corner = [center[i]+int(gradient_shape//2) if center[i]+int(
            gradient_shape//2) <= given_size[i] else given_size[i] for i in range(2)]

        # mask
        mask_left = [left_corner[i] - ori_left[i]   for i in range(len(left_corner))]
        mask_right = [right_corner[i] - ori_right[i] +  gradient_shape for i in range(len(right_corner))]
        result[left_corner[0]:right_corner[0], left_corner[1]:right_corner[1]
               ] = mask[mask_left[0]:mask_right[0], mask_left[1]:mask_right[1]]

        return result

