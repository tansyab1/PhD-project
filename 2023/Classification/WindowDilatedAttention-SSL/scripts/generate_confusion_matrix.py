import csv
import argparse
import numpy as np

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics

parser = argparse.ArgumentParser(
    description="Generate confusion matrix based on given prediction file.")

np.set_printoptions(linewidth=np.inf)

parser.add_argument("-i", "--input-prediction-file", type=str, required=True)
parser.add_argument("-o", "--output-file", type=str,
                    default="./confusion_matrix.pdf")

INDEX_TO_LETTER = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"
}

INDEX_TO_LABEL = {
    # 0:  "Ampulla of vater", deprecated
    0:  "Angiectasia",
    1:  "Blood - fresh",
    # 3:  "Blood - hematin", deprecated
    2:  "Erosion",
    3:  "Erythematous",
    # 6:  "Foreign Bodies", deprecated
    # 7:  "Ileo-cecal valve", deprecated
    4:  "Lymphangiectasia",
    5:  "Normal",
    6:  "Polyp",
    # 11:  "Pylorus", deprecated
    # 12:  "Reduced Mucosal View", deprecated
    7: "Ulcer",
}

LABEL_TO_LETTER = {
    # "Ampulla of vater": "A", deprecated
    "Angiectasia": "A",
    "Blood - fresh": "B",
    # "Blood - hematin": "D", deprecated
    "Erosion": "C",
    "Erythematous": "D",
    # "Foreign Bodies": "G", deprecated
    # "Ileo-cecal valve": "H", deprecated
    "Lymphangiectasia": "E",
    "Normal": "F",
    "Polyp": "G",
    # "Pylorus": "L", deprecated
    # "Reduced Mucosal View": "M", deprecated
    "Ulcer": "H"
}


def read_prediction_file(file_path, index_to_label=None):

    y_true = []
    y_pred = []

    with open(file_path) as csv_file:

        reader = csv.reader(csv_file, delimiter=",")

        next(reader)

        for row in reader:

            y_pred_value = row[1]
            y_true_value = row[2]
            print(y_true_value)

            if index_to_label is not None:
                y_true_value = index_to_label[int(y_true_value)]
                y_pred_value = index_to_label[int(y_pred_value)]

                y_true_value = LABEL_TO_LETTER[y_true_value]
                y_pred_value = LABEL_TO_LETTER[y_pred_value]

            y_true.append(y_true_value)
            y_pred.append(y_pred_value)

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, filename, labels, ymap=None, figsize=(15, 10)):

    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]

    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = "%.1f%%\n%d/%d" % (p, c, s)
            elif c == 0:
                annot[i, j] = ""
            else:
                annot[i, j] = "%.1f%%\n%d" % (p, c)

    cm = pd.DataFrame(cm_perc, index=labels, columns=labels)
    cm.index.name = "Actual"
    cm.columns.name = "Predicted"

    plt.rc("axes", labelsize=22)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)

    fig, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis='both', which='major', pad=10)

    sns.heatmap(cm, annot=annot, fmt="", ax=ax, cmap="Purples")

    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    ax.set_ylim(len(labels) + 0.5, -0.5)

    plt.savefig(filename, dpi=500)


if __name__ == "__main__":

    # args = parser.parse_args()

    input_prediction_file = '/home/nguyentansy/DATA/PhD-work/Datasets/output/ref-blur/train-0_val-1/evaluation.csv'
    # output_file = args.output_file

    y_true, y_pred = read_prediction_file(
        input_prediction_file, INDEX_TO_LABEL)

    plot_confusion_matrix(y_true, y_pred, "%s.png" % os.path.basename(
        input_prediction_file), sorted(list(INDEX_TO_LETTER.values())))
