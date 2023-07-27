# -*- coding:utf-8 -*-
import tensorflow as tf
# import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa


class Data_set(object):
    def __init__(self, config, shuffle, name, path):
        self.tfrecord_file = config.tfdata_path
        self.batch_size = config.batch_size
        self.min_after_dequeue = config.min_after_dequeue
        self.capacity = config.capacity
        self.actual_image_size = config.train_image_size
        self.shuffle = shuffle
        self.name = name

    # Set up the ImageDataGenerator with desired preprocessing
        self.datagen = ImageDataGenerator(
            rescale=1./255,
        )
        self.train_generator = self.datagen.flow_from_directory(
            path,
            target_size=(self.actual_image_size, self.actual_image_size),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            class_mode='categorical',
            seed=42
        )

    def next_iter(self):
        image_batch, label_batch = next(self.train_generator)
        # random rotate

        image_batch = tf.convert_to_tensor(image_batch)
        image_batch = tfa.image.rotate(
            image_batch,
            tf.random.uniform(shape=(tf.shape(image_batch)[0], ),
                              minval=-0.5,
                              maxval=0.5, seed=37),
            interpolation='BILINEAR')
        image_batch = dataaugmentation(image_batch)
        label_batch = tf.convert_to_tensor(label_batch)

        return image_batch, label_batch


def dataaugmentation(images):
    images = tf.image.random_flip_up_down(images)
    images = tf.image.random_flip_left_right(images)
    return images
