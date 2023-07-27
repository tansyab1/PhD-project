"""
Provide the images both in low and high resolution, 
the low res images are used as input1, saliency1 is used to sample the high res images to provide input2.
The dataset should be saved in the TF-record.
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# def get_image_label_batch(batch_size, which_split, shuffle, name):
#     with tf.name_scope('get_batch'):
#         Data = Data_set(batch_size=batch_size,
#                         which_split=which_split, shuffle=shuffle, name=name)
#         high_res_image_batch, low_res_image_batch, label_batch = \
#             Data.read_processing_generate_image_label_batch(batch_size)

#     return high_res_image_batch, low_res_image_batch, label_batch


class Data_set(object):
    def __init__(self, batch_size, which_split, shuffle, name, data_train):
        # self.data_train = "./downstream_folds/train/"
        # self.data_test = "./downstream_folds/test/"
        # self.data_val = "./downstream_folds/val/"
        self.data_train = data_train
        self.min_after_dequeue = 100
        self.capacity = 200
        self.high_res = 336
        self.low_res = 128
        self.shuffle = shuffle
        self.name = name

    # Set up the ImageDataGenerator with desired preprocessing
        self.datagen = ImageDataGenerator(
            rescale=1./255
        )
        self.train_generator = self.datagen.flow_from_directory(
            self.data_train,
            target_size=(self.high_res, self.high_res),
            batch_size=batch_size,
            shuffle=self.shuffle,
            class_mode='categorical',
        )

        # # Create the validation data generator
        # self.validation_generator = self.datagen.flow_from_directory(
        #     self.data_val,
        #     target_size=(self.high_res, self.high_res),
        #     batch_size=batch_size,
        #     class_mode='categorical',
        # )

    def next_iter(self):

        # if self.name == 'train':
        #     # get tensor of image/label
        #     image_batch, label_batch = next(self.train_generator)
        # else:
        #     image_batch, label_batch = next(self.validation_generator)
        image_batch, label_batch = next(self.train_generator)
        high_res_image_batch = image_batch
        low_res_image_batch = tf.image.resize(
            image_batch, [self.low_res, self.low_res])

        # high_res_image_batch = image_standardization(high_res_image_batch)
        # low_res_image_batch = image_standardization(low_res_image_batch)

        # print(high_res_image_batch, low_res_image_batch, label_batch)
        high_res_image_batch = tf.convert_to_tensor(high_res_image_batch)
        low_res_image_batch = tf.convert_to_tensor(low_res_image_batch)
        label_batch = tf.convert_to_tensor(label_batch)

        return high_res_image_batch, low_res_image_batch, label_batch


# def image_standardization(image):
#     out_image = image/255.0
#     return out_image
