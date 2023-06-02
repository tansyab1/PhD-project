# Imports
# %matplotlib inline
# %config InlineBackend.figure_formats = ['svg']

import cv2
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm  # Colormaps
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

sns.set_style('darkgrid')
np.random.seed(42)


def hybrid_multivariate_normal(x_p, x_q, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = np.matrix([[x_p], [np.log(x_q)]]) - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * (1/np.prod(x_q)) * np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))


def generate_hybrid_surface(mean, covariance, d):
    """Helper function to generate density surface."""
    nb_of_x = 772  # grid size
    x1s = np.linspace(-10, 10, num=nb_of_x)
    x2s = np.linspace(0.1, 20, num=nb_of_x)
    x1, x2 = np.meshgrid(x1s, x2s)  # Generate grid
    pdf = np.zeros((nb_of_x, nb_of_x))
    # Fill the cost matrix for each combination of weights
    for i in range(nb_of_x):
        for j in range(nb_of_x):
            pdf[i, j] = hybrid_multivariate_normal(
                x1[i, j], x2[i, j],
                d, mean, covariance)
    return x2, x1, pdf  # x1, x2, pdf(x1,x2)


# subplot
fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
d = 2  # number of dimensions

# Plot of independent Normals
bivariate_mean = np.matrix([[0], [2.5]])  # Mean
bivariate_covariance = np.matrix([
    [25, 0],
    [0, 0.5]])  # Covariance
x1, x2, p = generate_hybrid_surface(
    bivariate_mean, bivariate_covariance, d)

# show the output image
# ax1.contourf(x1, x2, p, 100, cmap='GnBu')

# # plot the side histogram of x1
# divider = make_axes_locatable(ax1)
# axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax1)
# axHistx.plot(x1[0, :], p[0, :], color='black', linewidth=1)
# axHistx.set_ylim([0, 0.1])
# axHistx.set_yticks([0, 0.05, 0.1])
# axHistx.set_yticklabels([0, 0.05, 0.1])
# axHistx.set_ylabel(r'$p(x_1)$', fontsize=12)
# axHistx.grid(False)

# # plot the side histogram of x2
# divider = make_axes_locatable(ax1)
# axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=ax1)
# axHisty.plot(p[:, 0], x2[:, 0], color='black', linewidth=1)
# axHisty.set_xlim([0, 0.1])
# axHisty.set_xticks([0, 0.05, 0.1])
# axHisty.set_xticklabels([0, 0.05, 0.1])
# axHisty.set_xlabel(r'$p(x_2)$', fontsize=12)
# axHisty.grid(False)


# plt.show()
# # save the image to eps
# # save folder path: 2023/Enhancement/eps/
# plt.savefig('hybrid.eps', format='eps', dpi=1000)

img = p.copy()
# img = np.where(np.isnan(img) == True, 0, p)
# norm = 255*(img - np.min(img)) / (np.max(img) - np.min(img))
norm = 255*(img) / (np.max(img))
(h, w) = norm.shape[:2]
(cX, cY) = (w // 2, h // 2)
# rotate our image by 45 degrees around the center of the image
M = cv2.getRotationMatrix2D((cX, cY), 90, 1.0)
rotated = cv2.warpAffine(norm, M, (w, h))

# crop 336x336 from the center
(h, w) = rotated.shape[:2]
(cX, cY) = (w // 2, h // 2)
rotated = rotated[cY-168:cY+168, cX-168-30:cX+168-30]

#  show the line of function like contourf
# ax1.plot(x1, x2, color='black', linewidth=1)

# flop the image
rotated = cv2.flip(rotated, 0)

# set figure size
fig.set_size_inches(4, 4)

# show the output image
# remove the grid
ax1.grid(False)
# tight layout
plt.tight_layout()
# plt.imshow(rotated, cmap='GnBu', vmin=0, vmax=255)
plt.imshow(rotated, cmap='gray', vmin=0, vmax=255)
plt.show()
# wait for 1 second
# cv2.waitKey()
# save the image
cv2.imwrite('90.png', rotated)
