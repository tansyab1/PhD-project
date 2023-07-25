"""
Provide the images both in low and high resolution, 
the low res images are used as input1, saliency1 is used to sample the high res images to provide input2.
The dataset should be saved in the TF-record.
"""
import tensorflow as tf


def get_image_label_batch(batch_size, which_split, shuffle, name):
    with tf.name_scope('get_batch'):
        Data = Data_set(batch_size=batch_size,
                        which_split=which_split, shuffle=shuffle, name=name)
        high_res_image_batch, low_res_image_batch, label_batch = \
            Data.read_processing_generate_image_label_batch(batch_size)

    return high_res_image_batch, low_res_image_batch, label_batch


class Data_set(object):
    def __init__(self, batch_size, which_split, shuffle, name):
        self.data_train = "./downstream_folds/train/"
        self.data_test = "./downstream_folds/test/"
        self.data_val = "./downstream_folds/val/"
        self.min_after_dequeue = 100
        self.capacity = 200
        self.high_res = 336
        self.low_res = 128
        self.shuffle = shuffle
        self.name = name

    def read_processing_generate_image_label_batch(self, batch_size):
        if self.name == 'train':
            # get tensor of image/label
            image_batch, label_batch = tf.keras.preprocessing.image_dataset_from_directory(
                self.data_train, labels='inferred', label_mode='int',
                class_names=None, color_mode='rgb', batch_size=batch_size,
                image_size=(self.high_res,
                            self.high_res), shuffle=self.shuffle,
                seed=None, validation_split=None, subset=None,
                interpolation='bilinear', follow_links=False)

        else:
            image_batch, label_batch = tf.keras.preprocessing.image_dataset_from_directory(
                self.data_val, labels='inferred', label_mode='int',
                class_names=None, color_mode='rgb', batch_size=batch_size,
                image_size=(self.high_res,
                            self.high_res), shuffle=self.shuffle,
                seed=None, validation_split=None, subset=None,
                interpolation='bilinear', follow_links=False)
            # num_threads=self.num_threads)
        high_res_image_batch = image_batch
        low_res_image_batch = tf.image.resize_images(
            image_batch, [self.low_res, self.low_res])

        high_res_image_batch = image_standardization(high_res_image_batch)
        low_res_image_batch = image_standardization(low_res_image_batch)

        return high_res_image_batch, low_res_image_batch, label_batch


def image_standardization(image):
    out_image = image/255.0
    return out_image
