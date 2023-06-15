import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


level = ['1', '2', '4', '8', '16', '32', '64']
# level = [1, 2, 3, 4, 5, 6, 7]
noise_Davies = [1.0837, 2.3689, 1.0257, 0.9932, 8.3121, 2.9187, 1.5019]
noise_Silhouette = [0.4136, 0.1425, 0.4317, 0.4394, 0.0086, 0.0663, 0.2424]

ui_Davies = [3.0059, 2.7809, 1.9159, 1.8255, 1.9255, 1.5229, 1.8640]
ui_Silhouette = [0.0984, 0.1079, 0.3065, 0.3038, 0.3053, 0.4065, 0.3891]

blur_Davies = [9.7781, 4.4354, 2.8199, 6.8242, 4.3365, 10.5562, 12.6697]
blur_Silhouette = [0.0089, 0.1136, 0.2536, 0.0346, 0.0851, 0.0058, 0.0043]
# plot the line chart with Davies-Bouldin index in line and Silhouette index in dot

# plt.plot(level, noise_Davies, color='blue', label='Davies-Bouldin (Noise)')
# plt.plot(level, noise_Silhouette, color='blue', linestyle='--', label='Silhouette (Noise)')
# plt.plot(level, ui_Davies, color='orange', label='Davies-Bouldin (UI)')
# plt.plot(level, ui_Silhouette, color='orange', linestyle='--', label='Silhouette (UI)')
# plt.plot(level, blur_Davies, color='green', label='Davies-Bouldin (Blur)')
# plt.plot(level, blur_Silhouette, color='green', linestyle='--', label='Silhouette (Blur)')

# # set the position of the x ticks
# plt.xticks(level)
# # set y label
# plt.ylabel('index')
# # set title of plot
# plt.title('Davies-Bouldin and Silhouette index')

# # legend on the top right of plot
# plt.legend(loc='upper right')

# # tight layout
# plt.tight_layout()

# # save the plot
# plt.savefig('line.png')

# plot two line charts in one figure with Davies-Bouldin index in line and Silhouette index in dot

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# text size of x and y ticks 15


ax1.plot(level, noise_Davies, color='blue', label='Davies-Bouldin (Noise)')
ax1.plot(level, ui_Davies, color='orange', label='Davies-Bouldin (UI)')
ax1.plot(level, blur_Davies, color='green', label='Davies-Bouldin (Blur)')
# ax1.set_xticks(stick)
ax1.set_ylabel('value')
ax1.set_xlabel('margin')
ax1.set_title('Davies-Bouldin index')
# put legend on the top right of plot ax1
ax1.legend(loc='upper left')
# put ax1 

ax2.plot(level, noise_Silhouette, color='blue', linestyle='--', label='Silhouette (Noise)')
ax2.plot(level, ui_Silhouette, color='orange', linestyle='--', label='Silhouette (UI)')
ax2.plot(level, blur_Silhouette, color='green', linestyle='--', label='Silhouette (Blur)')
# ax2.set_xticks(stick)
ax2.set_ylabel('value')
ax2.set_xlabel('margin')
ax2.set_title('Silhouette index')
ax2.legend(loc='upper left')

# fig.legend(loc='upper right')

plt.tight_layout()
# save the plot
plt.savefig('2023/src/index.eps', format='eps', dpi=1000)