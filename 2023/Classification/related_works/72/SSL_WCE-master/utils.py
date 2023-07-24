# -*- coding:utf-8 -*-
import scipy.misc
import numpy as np
import os
from glob import glob
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
#from keras.datasets import cifar10, mnist
from tensorflow.contrib.framework import arg_scope, add_arg_scope
from tensorflow.contrib.layers import batch_norm
from tflearn.layers.conv import global_avg_pool

import utilsForTF


def get_image_label_batch(config, shuffle, name):
    with tf.name_scope('get_batch'):
        Data = utilsForTF.Data_set(config, shuffle=shuffle, name=name)
        image_batch, label_batch = Data.read_processing_generate_image_label_batch()
    return image_batch, label_batch

def count_trainable_params():
    total_parameters = 0
    a = []
    for variable in tf.trainable_variables():
        a.append(variable)
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total training params: %.1fM" % (total_parameters / 1e6))
    return total_parameters

class ImageData:

    def __init__(self, load_size, channels):
        self.load_size = load_size
        self.channels = channels

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        return img


#def load_mnist(size=64):
#    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
#    train_data = normalize(train_data)
#    test_data = normalize(test_data)

#    x = np.concatenate((train_data, test_data), axis=0)
    # y = np.concatenate((train_labels, test_labels), axis=0).astype(np.int)

#    seed = 777
#    np.random.seed(seed)
#    np.random.shuffle(x)
    # np.random.seed(seed)
    # np.random.shuffle(y)
    # x = np.expand_dims(x, axis=-1)

#    x = np.asarray([scipy.misc.imresize(x_img, [size, size]) for x_img in x])
#    x = np.expand_dims(x, axis=-1)
#    return x

'''def load_cifar10(size=64) :
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    train_data = normalize(train_data)
    test_data = normalize(test_data)

    x = np.concatenate((train_data, test_data), axis=0)
    # y = np.concatenate((train_labels, test_labels), axis=0).astype(np.int)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(x)
    # np.random.seed(seed)
    # np.random.shuffle(y)

    x = np.asarray([scipy.misc.imresize(x_img, [size, size]) for x_img in x])

    return x

def load_data(dataset_name, size=64) :
    if dataset_name == 'mnist' :
        x = load_mnist(size)
    elif dataset_name == 'cifar10' :
        x = load_cifar10(size)
    else :

        x = glob(os.path.join("./dataset", dataset_name, '*.*'))

    return x'''


def preprocessing(x, size):
    x = scipy.misc.imread(x, mode='RGB')
    x = scipy.misc.imresize(x, [size, size])
    x = normalize(x)
    return x

def normalize(x) :
    return x/127.5 - 1

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def merge(images, size):
    images = np.nan_to_num(images)
    h, w = images.shape[1], images.shape[2]
    h, w = 128, 128
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = cv2.resize(image,(h,w))
            #img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = cv2.resize(image,(h,w))
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    # image = np.squeeze(merge(images, size)) # 채널이 1인거 제거 ?
    return scipy.misc.imsave(path, merge(images, size))


def inverse_transform(images):
    return (images+1.)/2.


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')

@add_arg_scope
def conv2layer(images, kernel, stride, output_channel, activation='relu', padding_mode='SAME', bn=False,
               trainning=False, scope='conv2'):
    with tf.variable_scope(scope):
        # input_channel = images.get_shape().as_list()[3]
        # weights = tf.get_variable("w",
        #                           shape=[kernel[0], kernel[1], input_channel, output_channel],
        #                           dtype=tf.float32,
        #                           initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        # biases = tf.get_variable("b",
        #                          shape=[output_channel],
        #                          dtype=tf.float32,
        #                          initializer=tf.constant_initializer(0.1))
        hidden = tf.layers.conv2d(images, output_channel, kernel, stride, padding_mode)
        if bn:
            hidden = bn2(images=hidden, training=trainning, scope='bn')
        if activation == 'relu':
            hidden = tf.nn.relu(hidden)
            return hidden
        else:
            return hidden
        # conv = tf.nn.conv2d(images, output_channel, strides=[1, stride, stride, 1], padding=padding_mode)
        # pre_activation = tf.nn.bias_add(conv, biases)
        # if activation == 'relu':
        #     return tf.nn.relu(pre_activation, name="relu")
        # else:
        #     return pre_activation

@add_arg_scope
def deconv2layer(images, kernel, stride, output_channel, activation='relu', padding_mode='SAME', bn=False, trainning=False, scope='coner2'):
    with tf.variable_scope(scope):
        # input_channel = images.get_shape().as_list()[3]

        hidden = tf.layers.conv2d_transpose(images, output_channel, kernel, stride, padding_mode)
        if bn:
            hidden = bn2(images=hidden, training=trainning, scope='bn')
        if activation == 'relu':
            hidden =  tf.nn.relu(hidden)
            return hidden
        else:
            return hidden

@add_arg_scope
def pool2layer(images, kernel, stride, pooling_mode='max', padding_mode='SAME', scope='pool2'):
    with tf.variable_scope(scope):
        if pooling_mode == 'max':
            return tf.nn.max_pool(images, ksize=[1, kernel[0], kernel[1], 1], strides=[1, stride, stride, 1],
                               padding=padding_mode, name="pooling1")
        else:
            return tf.nn.avg_pool(images, ksize=[1, kernel[0], kernel[1], 1], strides=[1, stride, stride, 1],
                                  padding=padding_mode, name="pooling1")
        
@add_arg_scope
def lrn(images,scope='lrn'):
    with tf.variable_scope(scope):
        return tf.nn.lrn(images, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                              beta=0.75, name='lrn')

@add_arg_scope
def bn2(images, training, scope):
    with tf.variable_scope(scope):
        return batch_norm(inputs=images,scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True, is_training=training)

@add_arg_scope
def PReLU(x, scope):
    # PReLU(x) = x if x > 0, alpha*x otherwise

    alpha = tf.get_variable(scope + "/alpha", shape=[1],
                initializer=tf.constant_initializer(0), dtype=tf.float32)

    output = tf.nn.relu(x) + alpha*(x - abs(x))*0.5

    return output

@add_arg_scope# function for 2D spatial dropout:
def spatial_dropout(x, drop_prob):
    # x is a tensor of shape [batch_size, height, width, channels]

    keep_prob = 1.0 - drop_prob
    input_shape = x.get_shape().as_list()

    batch_size = input_shape[0]
    channels = input_shape[3]

    # drop each channel with probability drop_prob:
    noise_shape = tf.constant(value=[batch_size, 1, 1, channels])
    x_drop = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape)

    output = x_drop

    return output