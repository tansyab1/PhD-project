import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import os


def plot_correlation_matrix(matrix, labels, add, name):
    # Create a figure with a single subplot
    fig, (ax, cax) = plt.subplots(2, 1, figsize=(6, 6))
    sns.heatmap(matrix,  cmap=sns.diverging_palette(20, 220, n=200),
                annot=True, fmt=".1f", ax=ax, cbar=False, mask=np.eye(len(matrix)))

    # Customize the axis labels
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation=0)

    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    plt.tight_layout()
    ax.set_facecolor('black')
    # plot second heatmap in horizontal direction
    # sort the add list
    add2 = np.sort(add, axis=1)

    # zip the labels and add
    zipadd = zip(labels, add[0])
    # sort the zipadd by add
    zipadd2 = sorted(zipadd, key=lambda x: x[1])

    # get the sorted labels
    sorted_labels = [x[0] for x in zipadd2]

    sns.heatmap(pd.DataFrame(add2), cmap=sns.diverging_palette(20, 220, n=200),
                annot=True, ax=cax, cbar=False, fmt=".1f")
    # print(zipadd2[0][1])
    cax.set_xticklabels(sorted_labels, rotation=90)
    cax.set_aspect('equal')
    cax.set_title('Total')
    # set cax figure size
    # cax.set_aspect(0.1)

    # rotate the heatmap
    cax.tick_params(axis='x', rotation=90)
    # remove y stick
    cax.yaxis.set_visible(False)
    # Add a title
    ax.set_title('Pairwise Comparison')

    # reduce the bottom margin of the cax

    # Adjust the layout and display the plot
    plt.tight_layout()
    # remove margin
    plt.subplots_adjust(bottom=-0.05)
    # plt.show()
    # save figure to eps file
    plt.savefig(os.path.join('2023/Enhancement/src/', name), format='eps')


# Example usage
matrix_blur = np.array([[0,	1,	1,	1,	1,	0,	0,	0],
                        [4,	0,	4,	0,	3.5, 1,	0,	0],
                        [4,	1,	0,	1,	1,	0,	0,	0],
                        [4,	5,	4,	0,	3,	1,	0,	0],
                        [4,	1.5, 4,	2,	0,	1,	0,	0],
                        [5,	4,	5,	4,	4,	0,	1,	1],
                        [5,	5,	5,	5,	5,	4,	0,	0],
                        [5,	5,	5,	5,	5,	4,	5,	0]])

method_blur = ['TV', 'DBGAN', 'D.GANv2', 'DMPHN',
               'MPRNet', 'MIRNet', 'Uformer', 'TCFA']

add_blur = [[4, 12.5, 7, 17, 12.5, 24, 29, 34]]


plot_correlation_matrix(matrix_blur, method_blur,
                        add_blur, name='blur_pairwise_node.eps')

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


matrix_noise = np.array([[0, 3, 3, 2, 3, 3, 2, 1, 0],
                         [2, 0, 3, 2, 2, 3, 0, 0, 0],
                         [2, 2, 0, 2, 2, 2, 1, 0, 0],
                         [3, 3, 3, 0, 2, 3, 1, 1, 0],
                         [2, 3, 3, 3, 0, 4, 0, 0, 0],
                         [2, 2, 3, 2, 1, 0, 0, 0, 0],
                         [3, 5, 4, 4, 5, 5, 0, 1, 0],
                         [4, 5, 5, 4, 5, 5, 4, 0, 0],
                         [5, 5, 5, 5, 5, 5, 5, 5, 0]])

# # sum of each row in matrix_noise
add_noise = [[17, 12, 11, 16, 15, 10, 27, 32, 40]]


plot_correlation_matrix(matrix_noise, method_noise, add_noise, name='noise_pairwise_node.eps')


method_ui = ['AFGT', 'FLM', 'LIME', 'Ret.Net',
             'En.GAN', 'MIRNet', 'FCN', 'Uformer', 'TCFA']

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


add_ui = [[12, 6, 8, 21, 39, 24, 33, 27, 46]]

plot_correlation_matrix(matrix_ui, method_ui, add_ui, name='ui_pairwise_node.eps')
