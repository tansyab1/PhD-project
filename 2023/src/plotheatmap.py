import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_correlation_matrix(matrix, labels):
    # Create a figure with a single subplot
    fig, (ax, cax) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the correlation matrix
    sns.heatmap(matrix,  cmap=sns.diverging_palette(20, 220, n=200), annot=True,fmt=".2f", ax=ax, cbar=True, mask=np.eye(len(matrix)))

    # Customize the axis labels
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels[::-1], rotation=0)

    # Move the y-axis ticks to the top
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    # equal aspect ratio ensures that cells are square-shaped
    ax.set_aspect('equal')
    np.fill_diagonal(matrix, np.nan)
    # fill diagonal of matrix with black color
    # ax.fill_diagonal(matrix, np.nan)
    ax.set_facecolor('black')
    extra_column = pd.Series(np.random.rand(5), name='Extra')
    sns.heatmap(pd.DataFrame(extra_column), cmap='coolwarm', annot=True, ax=cax, cbar=False)
    cax.set_title('Additional Column')

    # Add a title
    ax.set_title('Correlation Matrix')

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()

-	4	4	3
2	-	3	2
2	3	-	3
3	4	3	-
2	4	4	4
3	3	4	3
4	6	5	5
5	6	6	5
6	6	6	6

# Example usage
matrix = np.array([[1, 0.8, 0.6],
                   [0.8, 1.0, 0.4],
                   [0.6, 0.4, 1.0]])
methodnoise = ['BM3D', 'CycleISP', 'DANet']

plot_correlation_matrix(matrix, methodnoise)
