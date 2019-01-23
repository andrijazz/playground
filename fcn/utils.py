# !/usr/bin/python

import numpy as np
import scipy.misc
import tensorflow as tf
import os
import glob
import sys
import logging
import datetime


def make_hparam_string(config):
    separator = "_"
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    config_list = [now, config.model, str(config.epochs),
                   str(config.batch_size), str(config.learning_rate),
                   str(config.keep_prob)]
    hparams = separator.join(config_list)
    return hparams


def setup_logger(logger_name, log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_file = logger_name
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s', datefmt='%d-%b-%y-%H:%M:%S')

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler("{0}/{1}.log".format(log_path, log_file))
    c_handler.setLevel(logging.DEBUG)
    f_handler.setLevel(logging.DEBUG)

    # Add formatter to handlers
    c_handler.setFormatter(log_formatter)
    f_handler.setFormatter(log_formatter)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def load_data(images_path, gt_images_path):
    images = glob.glob(os.path.join(images_path, '*.png'))
    data_set = []
    for image in images:
        filename = os.path.basename(image)
        gt_image = gt_images_path + "/" + filename
        if not os.path.exists(gt_image):
            print("Missing label for instance {}".format(filename))
            continue
        data_set.append([image, gt_image])
    return data_set


def calculate_min_image_size(images):
    min_height = sys.maxsize
    min_width = sys.maxsize
    for [x, y] in images:
        image = scipy.misc.imread(x)
        gt_image = scipy.misc.imread(y)

        if image.shape[0] < min_height:
            min_height = image.shape[0]

        if image.shape[1] < min_width:
            min_width = image.shape[1]

        if gt_image.shape[0] < min_height:
            min_height = gt_image.shape[0]

        if gt_image.shape[1] < min_width:
            min_width = gt_image.shape[1]

    return min_height, min_width


def center_crop(img, h, w):
    h_offset = (img.shape[0] - h) // 2
    w_offset = (img.shape[1] - w) // 2
    return img[h_offset : h_offset + h, w_offset : w_offset + w, :]


def adjust_images(images, min_height, min_width):
    """Crops images to min_height, min_width"""
    for [x, y] in images:
        image = scipy.misc.imread(x)
        cropped_image = center_crop(image, min_height, min_width)
        scipy.misc.imsave(x, cropped_image)
        gt_image = scipy.misc.imread(y)
        cropped_gt_image = center_crop(gt_image, min_height, min_width)
        scipy.misc.imsave(y, cropped_gt_image)


def load_samples(m, images, image_height, image_width):
    """Iterates through list of images and packs them into batch of size m"""
    x_batch = np.empty([m, image_height, image_width, 3])
    y_batch = np.empty([m, image_height, image_width, 3])
    for i in range(m):
        image_file = images[i][0]
        gt_image_file = images[i][1]
        image = scipy.misc.imread(image_file)
        gt_image = scipy.misc.imread(gt_image_file)
        x_batch[i, :, :, :] = image[0 : image_height, 0 : image_width, :]
        y_batch[i, :, :, :] = gt_image
    return x_batch, y_batch


def load_samples_prob(m, images, image_height, image_width, probability_classes):
    """Iterates through list of images and packs them into batch of size m"""
    x_batch = np.empty([m, image_height, image_width, 3])
    num_classes = len(probability_classes)
    y_batch = np.empty([m, image_height, image_width, 3])
    y_prob_batch = np.empty([m, image_height, image_width, num_classes])
    for i in range(m):
        image_file = images[i][0]
        gt_image_file = images[i][1]
        image = scipy.misc.imread(image_file)
        gt_image = scipy.misc.imread(gt_image_file)
        x_batch[i, :, :, :] = image[0 : image_height, 0 : image_width, :]
        y_batch[i, :, :, :] = gt_image[0: image_height, 0: image_width, :]
        y_prob_batch[i, :, :, :] = image2cprob(gt_image, probability_classes)
    return x_batch, y_batch, y_prob_batch


def load_samples_idx(m, images, image_height, image_width, idx_classes):
    """Iterates through list of images and packs them into batch of size m"""
    x_batch = np.empty([m, image_height, image_width, 3])
    y_batch = np.empty([m, image_height, image_width, 3])
    y_idx_batch = np.empty([m, image_height, image_width])
    for i in range(m):
        image_file = images[i][0]
        gt_image_file = images[i][1]
        image = scipy.misc.imread(image_file)
        gt_image = scipy.misc.imread(gt_image_file)
        x_batch[i, :, :, :] = image[0 : image_height, 0 : image_width, :]
        y_batch[i, :, :, :] = gt_image[0: image_height, 0: image_width, :]
        y_idx_batch[i, :, :] = image2cidx(gt_image, idx_classes)
    return x_batch, y_batch, y_idx_batch


def crop_tensor(x, h, w):
    # cropping the output image to original size
    # https://github.com/shelhamer/fcn.berkeleyvision.org/edit/master/README.md#L72
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/voc-fcn32s/net.py#L62
    offset_height = (tf.shape(x)[1] - h) // 2
    offset_width = (tf.shape(x)[2] - w) // 2
    return tf.image.crop_to_bounding_box(x, offset_height, offset_width, h, w)


def assign_variable(name, value):
    [scope, var] = name.split('/')
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        return tf.get_variable(var).assign(value)


# initialize weights from a truncated normal distribution
def weight_variable(shape, name):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name, shape, initializer=initial)


# initialize biases to constant values
def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape)
    return tf.get_variable(name, initializer=initial)


# 2d convolution with padding
# https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
def conv2d(x, W, s, padding):
    return tf.nn.conv2d(x, W, s, padding=padding)


# 2x2 max pooling with 2x2 stride and same padding
def max_pool_2x2(name, x):
    with tf.variable_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# convolution layer (shape = [filter_height, filter_width, in_channels, out_channels])
def conv_layer(name, shape, stride, relu, batch_norm, dropout, padding, x, keep_prob=0.5):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W = weight_variable(shape, 'weights')
        b = bias_variable([shape[3]], 'biases')
        h_conv = conv2d(x, W, stride, padding) + b

        if batch_norm:
            h_conv = tf.contrib.layers.batch_norm(h_conv)
        if relu:
            h_conv = tf.nn.relu(h_conv)
        if dropout:
            # setting keep prob to 0.5 according to
            # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/voc-fcn32s/net.py#L53
            h_conv = tf.nn.dropout(h_conv, keep_prob=keep_prob)

        return h_conv


# bilinear tensor upsampling
def upsample(x, s):
    shape = tf.shape(x)
    x_up = tf.image.resize_images(x, [shape[1] * s, shape[2] * s])
    return x_up


# https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers
def deconv_layer(name, n_channels, kernel_size, stride, x):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_shape = tf.shape(x)

        h = in_shape[1] * stride
        w = in_shape[2] * stride
        new_shape = [in_shape[0], h, w, n_channels]

        output_shape = tf.stack(new_shape)

        filter_shape = [kernel_size, kernel_size, n_channels, n_channels]

        W = weight_variable(filter_shape, 'weights')
        deconv = tf.nn.conv2d_transpose(x, W, output_shape, strides=strides, padding='SAME', name=name)
    return deconv


def image2cidx(img, idx_classes):
    image_height = img.shape[0]
    image_width = img.shape[1]
    cidx_img = np.zeros([image_height, image_width])
    for c in idx_classes:
        coords = np.where(img == np.array(c))
        cidx_img[coords[0][::3], coords[1][::3]] = idx_classes[c]
    return cidx_img


def image2cprob(img, probability_classes):
    image_height = img.shape[0]
    image_width = img.shape[1]
    num_classes = len(probability_classes)
    cprob_img = np.zeros([image_height, image_width, num_classes])
    for c in probability_classes:
        coords = np.where(img == np.array(c))
        cprob_img[coords[0][::3], coords[1][::3], :] = probability_classes[c]
    return cprob_img


def save_images(images, names, result_path):
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    m = images.shape[0]
    for i in range(m):
        basename = os.path.basename(names[i][0])
        scipy.misc.imsave(result_path + "/" + basename, images[i])


def collect(images_path):
    image_files = glob.glob(os.path.join(images_path, '*.png'))
    classes = set()
    for image_file in image_files:
        image = scipy.misc.imread(image_file)
        for px in range(image.shape[0]):
            for py in range(image.shape[1]):
                classes.add(tuple(image[px, py]))
    return classes


def check_gpu():
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
    with tf.Session() as sess:
        print(sess.run(c))
