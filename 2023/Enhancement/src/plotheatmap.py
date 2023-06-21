import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec


def plot_correlation_matrix(matrix, labels, add):
    # Create a figure with a single subplot
    fig, (ax, cax) = plt.subplots(1, 2, figsize=(10, 5))
    # gs1 = gridspec.GridSpec(2, 1)
    # gs1.update(wspace=025, hspace=05)
    # Plot the correlation matrix
    sns.heatmap(matrix,  cmap=sns.diverging_palette(20, 220, n=200),
                annot=True, fmt=".1f", ax=ax, mask=np.eye(len(matrix)))

    # Customize the axis labels
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation=0)

    # Move the y-axis ticks to the top
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    # set x label in the inverse order
    # ax.invert_yaxis()
    plt.tight_layout()
    # equal aspect ratio ensures that cells are square-shaped
    # ax.set_aspect('equal')
    # np.fill_diagonal(matrix, np.nan)
    # fill diagonal of matrix with black color
    # ax.fill_diagonal(matrix, np.nan)
    ax.set_facecolor('black')
    sns.heatmap(pd.DataFrame(add), cmap='coolwarm',
                annot=True, ax=cax, cbar=True, fmt=".1f")
    cax.set_yticklabels(labels, rotation=0)
    cax.set_aspect('equal')
    cax.set_title('Total')

    # Add a title
    ax.set_title('Pairwise Comparison')

    # Adjust the layout and display the plot
    plt.tight_layout()
    # save figure to eps file
    plt.savefig('2023/Enhancement/src/ui_pairwise_node.eps', format='eps')


# Example usage
# matrix_blur = np.array([[0,	1,	1,	1,	1,	0,	0,	0],
#                         [4,	0,	4,	0,	3.5, 1,	0,	0],
#                         [4,	1,	0,	1,	1,	0,	0,	0],
#                         [4,	5,	4,	0,	3,	1,	0,	0],
#                         [4,	1.5, 4,	2,	0,	1,	0,	0],
#                         [5,	4,	5,	4,	4,	0,	1,	1],
#                         [5,	5,	5,	5,	5,	4,	0,	0],
#                         [5,	5,	5,	5,	5,	4,	5,	0]])

# method_blur = ['TV', 'DBGAN', 'DeblurGANv2', 'DMPHN',
#           'MPRNet', 'MIRNet', 'Uformer', 'TCFA']

# add_blur = [4, 12.5, 7, 17, 12.5, 24, 29, 34]


# plot_correlation_matrix(matrix_blur, method_blur, add_blur)

method_noise = ['BM3D', 'CycleISP', 'DANet', 'DIPNet',
                'VDNet', 'MPRNet', 'MIRNet', 'Uformer', 'TCFA']

# 0 3 3 2 3 3 2 1 0
# 2 0 3 2 2 3 0 0 0
# 2 2 0 2 2 2 1 0 0
# 3 3 3 0 2 3 1 1 0
# 2 3 3 3 0 4 0 0 0
# 2 2 3 2 1 0 0 0 0
# 3 5 4 4 5 5 0 1 0
# 4 5 5 4 5 5 4 0 0
# 5 5 5 5 5 5 5 5 0


# matrix_noise = np.array([[0, 3, 3, 2, 3, 3, 2, 1, 0],
#                          [2, 0, 3, 2, 2, 3, 0, 0, 0],
#                          [2, 2, 0, 2, 2, 2, 1, 0, 0],
#                          [3, 3, 3, 0, 2, 3, 1, 1, 0],
#                          [2, 3, 3, 3, 0, 4, 0, 0, 0],
#                          [2, 2, 3, 2, 1, 0, 0, 0, 0],
#                          [3, 5, 4, 4, 5, 5, 0, 1, 0],
#                          [4, 5, 5, 4, 5, 5, 4, 0, 0],
#                          [5, 5, 5, 5, 5, 5, 5, 5, 0]])

# # sum of each row in matrix_noise
# add_noise = [17, 12, 11, 16, 15, 10, 27, 32, 40]


# plot_correlation_matrix(matrix_noise, method_noise, add_noise)


method_ui = ['AFGT', 'FLM', 'LIME', 'RetinexNet',
             'EnlightenGAN', 'MIRNet', 'FCN', 'Uformer', 'TCFA']

# 0 4 3 2 0 1 1 1 0
# 2 0 3 0 0 1 0 0 0
# 3 3 0 1 0 1 0 0 0
# 4 6 5 0 1 2 1 2 0
# 6 5 6 5 0 5 4 5 2
# 5 5 5 4 1 0 2 2 0
# 5 6 6 5 2 4 0 5 0
# 5 6 6 4 1 4 1 0 0
# 6 6 6 6 4 6 6 6 0


matrix_ui = np.array([[0, 4, 3, 2, 0, 1, 1, 1, 0],
                      [2, 0, 3, 0, 0, 1, 0, 0, 0],
                      [3, 3, 0, 1, 0, 1, 0, 0, 0],
                      [4, 6, 5, 0, 1, 2, 1, 2, 0],
                      [6, 6, 6, 5, 0, 5, 4, 5, 2],
                      [5, 5, 5, 4, 1, 0, 2, 2, 0],
                      [5, 6, 6, 5, 2, 4, 0, 5, 0],
                      [5, 6, 6, 4, 1, 4, 1, 0, 0],
                      [6, 6, 6, 6, 4, 6, 6, 6, 0]])


add_ui = [12, 6, 8, 21, 39, 24, 33, 27, 46]

plot_correlation_matrix(matrix_ui, method_ui, add_ui)
