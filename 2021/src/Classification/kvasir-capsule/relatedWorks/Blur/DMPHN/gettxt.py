# define a function to get the image path from a folder

import os
def get_image_path(folder):
    image_path = []
    # dirs_path = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".jpg"):
                image_path.append(os.path.join(root, file))
                # dirs_path.append(os.path.join(dirs))
    return image_path

if __name__ == "__main__":
    # get the image path
    blur_image_path = get_image_path("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Blur_var/test/input")
    sharp_image_path = get_image_path("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/Blur_var/test/groundtruth")
    # save the image path to a txt file
    with open("blur.txt", "w") as f:
        for path in blur_image_path:
            # write the image path to the txt file from Blur_var/test/input
            f.write(path.replace("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/", "") + "\n")
            
    with open("sharp.txt", "w") as f:
        for path in sharp_image_path:
            # write the image path to the txt file from Blur_var/test/groundtruth
            f.write(path.replace("/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/", "") + "\n")