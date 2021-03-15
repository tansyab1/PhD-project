import numpy as np
import cv2

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    cv2.imshow('re',dist_from_center)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    mask = dist_from_center <= radius
    return mask

if __name__ == '__main__':
    mask=create_circular_mask(100,100)
