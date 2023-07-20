# write a python file to read the .csv file and save the images in the folder

import os
import pandas as pd
from tqdm import tqdm


def readcsv(filename, foldername, destination_foldername):
    df = pd.read_csv(filename)
    # csv file has the comluumns: filename,predicted-label,A,B,C,D,E,F,G,H
    # if the predicted-label is not 5, then the image is saved in the folder with the name of the predicted-label

    for i in tqdm(range(len(df))):  # len(df) is the number of rows in the csv file
        if df['predicted-label'][i] != 5:
            # create a folder with the name of the predicted-label if it does not exist
            os.system('mkdir -p ' + destination_foldername +
                      str(df['predicted-label'][i]))
            # copy the image from the foldername to the folder with the name of the predicted-label
            os.system('cp ' + foldername + df['filename'][i] + ' ' + destination_foldername + str(
                df['predicted-label'][i]) + '/' + df['filename'][i])


# function to read all csv files and find the images with the predicted-label is 5 and copy line by line to a new csv file with the name of the predicted-label
def readcsv2(foldername, csv_file_list, file_destination):
    for csv_file in csv_file_list:
        df_temp = pd.read_csv(csv_file)
        # select random 1000 images with the predicted-label is 5
        df_temp = df_temp[df_temp['predicted-label'] == 5].sample(n=10000)

        # read each row line in df_temp and copy the image from the foldername to the folder with the name of the predicted-label
        for i in tqdm(range(len(df_temp))):
            # create a folder with the name of the predicted-label if it does not exist
            os.system('mkdir -p ' + file_destination +
                      str(df_temp['predicted-label'].iloc[i]))
            # copy the image from the foldername to the folder with the name of the predicted-label
            os.system('cp ' + foldername + df_temp['filename'].iloc[i] + ' ' + file_destination + str(
                df_temp['predicted-label'].iloc[i]) + '/' + df_temp['filename'].iloc[i])


if __name__ == '__main__':
    # # csv_file = '2023/Classification/Pre-Trained-DenseNet-161/result/output/densenet161_split_0.py_inference_fold_1.csv'
    # folder_name = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/unlabelled_images/pseudo/'
    # destination_folder_name = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/unlabelled_images/pseudo_labelled/'
    # # readcsv(csv_file, folder_name, destination_folder_name)
    csv_list = []
    for i in tqdm(range(0, 10)):
        csv_file = '2023/Classification/Pre-Trained-DenseNet-161/result/output/densenet161_split_0.py_inference_fold_' + \
            str(i) + '.csv'
        csv_list.append(csv_file)
    csv_file_destination = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/unlabelled_images/pseudo_labelled/'
    folder_name = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/unlabelled_images/pseudo/'
    readcsv2(folder_name, csv_list, csv_file_destination)
