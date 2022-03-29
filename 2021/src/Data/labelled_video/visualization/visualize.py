import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import glob
from tqdm import tqdm
import os

for file in tqdm(glob.glob("src/Pre-processing/Isotropic/KvasirCapsule/coefs/*.csv")):
    orinames=os.path.splitext(os.path.basename(file))[0]
    df_data = pd.read_csv(file, delimiter=',', header=None)

    # estimated coefficients of different models
    df_data.columns = ['ui', 'noise', 'blur']

    data_ui = df_data.values[:, 0]
    data_noise = df_data.values[:, 1]
    data_blur = df_data.values[:, 2]

    t = np.arange(0, data_noise.shape[0], 1)
    # plt.plot(t, data_ui/max(data_ui), 'r--', t, data_noise/max(data_noise), 'bs', t, 100*max(data_noise)/data_blur, 'g^')
    # plt.show()

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
    plt.plot(t, data_blur/max(data_blur), color='tab:green', marker='o')
    plt.plot(t, data_blur/max(data_blur), color='black')
    plt.xticks(np.arange(0, len(t)+1, 100))
    plt.legend(["Blur"])
    print("===============================")
    print(orinames)
    plt.suptitle(orinames)
    # manager = plt.get_current_fig_manager()
    # manager.frame.Maximize(True)
    # plt.savefig("src/Pre-processing/Isotropic/KvasirCapsule/coefs/visualization/"+orinames+".png",bbox_inches='tight')
    plt.show()
    # plt.savefig("src/Pre-processing/Isotropic/KvasirCapsule/coefs/visualization/"+orinames+".png",bbox_inches='tight')
    