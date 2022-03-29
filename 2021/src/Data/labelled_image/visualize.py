import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import os

nums=[]
names=[]
for file in tqdm(glob.glob("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/*.csv")):
    orinames=os.path.splitext(os.path.basename(file))[0]
    df_data = pd.read_csv(file, delimiter='\t', header=None)

    # estimated coefficients of different models
    df_data.columns = ['ui', 'noise', 'blur']

    data_ui = df_data.values[:, 0]
    data_noise = df_data.values[:, 1]
    data_blur = df_data.values[:, 2]

    t = np.arange(0, data_noise.shape[0], 1)

    plt.figure(orinames)
    plt.subplot(311)
    plt.plot(t, data_ui/max(data_ui), color='tab:blue', marker='o')
    plt.plot(t, data_ui/max(data_ui), color='black')
    plt.xticks(np.arange(0, len(t)+1, 100))
    plt.legend(["UI"])

    plt.subplot(312)
    plt.plot(t, data_noise/max(data_noise), color='tab:orange', marker='o')
    plt.plot(t, data_noise/max(data_noise), color='black')
    plt.xticks(np.arange(0, len(t)+1, 100))
    plt.legend(["Noise"])

    plt.subplot(313)
    plt.plot(t, 1/(max(1/data_blur)*data_blur), color='tab:green', marker='o')
    plt.plot(t, 1/(max(1/data_blur)*data_blur), color='black')
    plt.xticks(np.arange(0, len(t)+1, 100))
    plt.legend(["Blur"])
    print("===============================")
    print(orinames)
    plt.suptitle(orinames)
    plt.show()

#     num = input("get number:")
#     nums.append(len(t)-int(num))

#     names.append(orinames)

# # write to csv
# df = pd.DataFrame({"name":names, "num":nums})
# df.to_csv("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/label.csv", index=False)





