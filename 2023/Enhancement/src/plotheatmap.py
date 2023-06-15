import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec

def plot_correlation_matrix(matrix, labels, add):
    # Create a figure with a single subplot
    fig, (ax, cax) = plt.subplots(1, 2, figsize=(10, 5))
    # gs1 = gridspec.GridSpec(2, 1)
    # gs1.update(wspace=0.025, hspace=0.05)
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
    plt.savefig('blur_pairwise.eps', format='eps')


					
# Example usage
matrix = np.array([[0,	2,	1,	1,	1,	1,	0,	0],
                   [4,	0,	4,	0,	3.5,	1,	0,	0],
                   [5,	2,	0,	1,	2,	1,	0,	0],
                   [5,	6,	5,	0,	4,	2,	1,	0],
                   [5,	4.5,	4,	2,	0,	2,	0,	0],
                   [5,	5,	5,	4,	4,	0,	1,	1],
                   [6,	6,	6,	5,	6,	5,	0,	0],
                   [6,	6,	6,	6,	6,	5,	6,	0]])
                   
method = ['TV', 'DBGAN', 'DeblurGANv2', 'DMPHN',
           'MPRNet', 'MIRNet', 'Uformer', 'TCFA']

add = [6, 10.5, 11, 23, 17.5, 25, 34, 41]


plot_correlation_matrix(matrix, method, add)
