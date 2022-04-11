from cProfile import label
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import seaborn as sns

# Import Data
df_raw = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")

names_lum = [
         "Angiectasia",
         "Blood - fresh",
         "Blood - hematin",
         "Erosion",
         "Erythema",
         "Foreign body",
         "Lymphangiectasia",
         "Normal clean mucosa",
         "Polyp",
         "Reduced mucosal view",
         "Ulcer"]

names_ana = ["Ampulla of vater",
            "Ileocecal valve",
            "Pylorus"]
input_dir = "/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/*/"


# count number of images
count_ana = []
count_lum = []
for name in names_ana:
    count_ana.append(len(glob.glob(input_dir+name+"/*.jpg")))

for name in names_lum:
    count_lum.append(len(glob.glob(input_dir+name+"/*.jpg")))

# Figure Size
fig, (ax) = plt.subplots()

# Horizontal Bar Plot
ax.barh(names_ana, count_ana, color='orange', align='center', alpha=0.5, edgecolor='black', label='Anatomy') 
ax.barh(names_lum, count_lum, color='blue', align='center', alpha=0.5, edgecolor='black', label='Luminal' )
ax.set_title('Number of Images', position=(0.5, 1.05), fontsize=14)
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
	ax.spines[s].set_visible(False)

# Remove x, y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)

# Add x, y gridlines
ax.grid(b = True, color ='grey',
		linestyle ='-.', linewidth = 0.5,
		alpha = 0.2)

# Show top values
ax.invert_yaxis()

# Add annotation to bars
for i in ax.patches:
	plt.text(i.get_width()+0.2, i.get_y()+0.5,
			str(round((i.get_width()), 2)),
			fontsize = 10, fontweight ='bold',
			color ='grey')

# Add Plot Title
ax.set_title('The number of images per category', fontsize=14,loc ='center', )

ax.legend(prop={'size': 14})
# Show Plot
plt.show()

