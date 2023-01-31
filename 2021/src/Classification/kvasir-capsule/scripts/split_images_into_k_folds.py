import glob
import shutil
import os
import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description="Split data into k folds.")

file_dir = os.path.dirname(os.path.realpath(__file__))

parser.add_argument("-s", "--src-dir", type=str,
                    default="/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/distorted_images/UI_var/")
parser.add_argument("-d", "--dest-dir", type=str,
                    default="/home/nguyentansy/DATA/PhD-work/Datasets/kvasir_capsule/labelled_images/process/labelled_images/ExperimentalDATA/forRelatedWorks/UI_var")
parser.add_argument("-f", "--number-of-folds", type=int, default=3)
parser.add_argument("-e", "--exclude-classes", nargs='+', default=[])


def split_data_into_equal_parts(data, number_of_parts):
    # suffle data
    data = np.random.permutation(data)
    part_length = [20000, 16000, 2000]
    parts = []

    parts += [data[0: part_length[0]]]
    parts += [data[part_length[0]: part_length[0] + part_length[1]]]
    parts += [data[part_length[0] + part_length[1]: part_length[0] + part_length[1] + part_length[2]]]
    return parts


def split_images(src_dir, number_of_folds, dest_dir=None, exclude_classes=[]):

    split_file = open("%s_fold_split_UI_var.csv" % str(number_of_folds), "w")

    split_file.write("file-name;class-name;split-index\n")
    split_index = 0

    for class_path in glob.glob("%s/*" % src_dir):

        class_name = os.path.basename(class_path)
        file_paths = list(glob.glob("%s/*" % class_path))
        # print(len(file_paths))

        if class_name in exclude_classes:
            print("Skipping class %s" % class_name)
            continue

        for split in split_data_into_equal_parts(file_paths, number_of_folds):
            # print(len(split))
            for file_path in tqdm(split):
                
                file_name = os.path.basename(file_path)

                split_file.write("%s;%s;%s\n" %
                                 (file_name, class_name, str(split_index)))

                if dest_dir is not None:

                    dest_class_path = os.path.join(dest_dir, str(split_index))

                    if not os.path.exists(dest_class_path):
                        os.makedirs(dest_class_path)

                    shutil.copy(file_path, os.path.join(
                        dest_class_path, file_name))
            
            split_index += 1


if __name__ == "__main__":

    args = parser.parse_args()

    src_dir = args.src_dir
    dest_dir = args.dest_dir
    number_of_folds = args.number_of_folds
    exclude_classes = args.exclude_classes

    if not dest_dir is None and os.path.exists(dest_dir):
        raise Exception(
            "%s already exists. Please delete it or choose another destination." % dest_dir)

    split_images(src_dir, number_of_folds, dest_dir, exclude_classes)
