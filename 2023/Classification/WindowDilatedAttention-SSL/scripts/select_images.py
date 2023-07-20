import os
import pandas as pd


# define a function to read folder and give the number of images in each folder
def read_folder(folder_name):
    folder_name_list = []
    num_images_list = []

    largest_folder_list = []
    largest_num_images_list = []
    for i in range(0, 8):
        # add the folder name and the number of images in the folder to the list
        if len(os.listdir(folder_name + str(i) + '/')) <= 100000:
            folder_name_list.append(folder_name + str(i) + '/')
            num_images_list.append(len(os.listdir(folder_name + str(i) + '/')))
            print(folder_name + str(i) + '/' + str(len(os.listdir(folder_name + str(i) + '/'))))
        else:
            largest_folder_list.append(i)
            largest_num_images_list.append(
                len(os.listdir(folder_name + str(i) + '/')))

    # zip the two lists together and sort by the number of images in the folder
    finalzip = sorted(zip(folder_name_list, num_images_list),
                      key=lambda x: x[1])

    # sum the number of images in the finalzip
    sum_images = sum([x[1] for x in finalzip])
    sum_images_largest = (400000 - sum_images)//len(largest_folder_list)

    # copy randomly the images from the largest folder to  another folder with sum_images_largest images
    os.system('mkdir -p ' + folder_name + 'largest/')
    for i in (largest_folder_list):
        # select random sum_images_largest in each folder of largest_folder_list
        selected_images = pd.Series(os.listdir(
            folder_name + str(i) + '/')).sample(n=sum_images_largest)
        # copy the selected images to the folder largest corresponding to the folder i
        os.system('mkdir -p ' + folder_name + 'largest/' + str(i) + '/')
        for selected_images in selected_images:
            os.system('cp ' + folder_name + str(i) + '/' + selected_images + ' ' +
                      folder_name + 'largest/' + str(i) + '/' + selected_images)


if __name__ == "__main__":
    folder_name = '/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/unlabelled_images/pseudo_labelled/'
    read_folder(folder_name)
