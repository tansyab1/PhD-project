import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Generate random correlation matrix
np.random.seed(0)
corr_matrix = pd.DataFrame(np.random.rand(5, 5), columns=['A', 'B', 'C', 'D', 'E'])

# Add an additional column to the right
extra_column = pd.Series(np.random.rand(5), name='Extra')
# corr_matrix = pd.concat([corr_matrix, extra_column], axis=1)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(ncols=2)
# set the size of ax2   
ax2.set_aspect('equal')
ax1.set_aspect('equal')
# put stick of x axis of ax1 on the top of ax1
ax1.xaxis.tick_top()

# change y label 


# Plot the correlation matrix
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, ax=ax1, cbar=False)
ax1.set_title('Correlation Matrix')

# Plot the additional column
sns.heatmap(pd.DataFrame(extra_column), cmap='coolwarm', annot=True, ax=ax2, cbar=False)
ax2.set_title('Additional Column')

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()