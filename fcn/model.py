from __future__ import absolute_import, division, print_function
from collections import namedtuple

import tensorflow as tf
import numpy as np
# TODO replace with slim
import vapk as utils


fcn_parameters = namedtuple('parameters',
                        'image_height, '
                        'image_width, '
                        'labels,'
                        'num_labels,'
                        'keep_prob, '
                        'learning_rate')


class fcn32(object):
    """fcn32 model"""

    def __init__(self, params, reuse_variables=None):
        self.params = params
        self.reuse_variables = reuse_variables
        self.__build_model()
        self.__build_loss()
        self.__build_summaries()
        # init id_to_color
        self.id_to_rgb = np.zeros((self.params.num_labels, 3))
        for label in self.params.labels:
            self.id_to_rgb[label.id] = label.color

    def __build_model(self):
        with tf.variable_scope('input', reuse=self.reuse_variables):
            self.x = tf.placeholder(tf.float32, shape=[None, self.params.image_height, self.params.image_width, 3], name="x")  # [batch, in_height, in_width, in_channels]
            self.y = tf.placeholder(tf.float32, shape=[None, self.params.image_height, self.params.image_width, 3], name="y")  # [batch, in_height, in_width, in_channels]
            self.y_probability = tf.py_func(self.batch_rgb_to_probability, [self.y], tf.float32)

            # padding input by 100
            # https://github.com/shelhamer/fcn.berkeleyvision.org/edit/master/README.md#L72
            # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/voc-fcn32s/net.py#L28
            padded_x = tf.pad(self.x, [[0, 0], [100, 100], [100, 100], [0, 0]], "CONSTANT")

        with tf.variable_scope('vgg-16', reuse=self.reuse_variables):
            h_conv1_1 = utils.conv_layer('conv1_1', [3, 3, 3, 64], [1, 1, 1, 1], True, False, False, "SAME",  padded_x)
            h_conv1_2 = utils.conv_layer('conv1_2', [3, 3, 64, 64], [1, 1, 1, 1], True, False, False, "SAME", h_conv1_1)
            h_pool1 = utils.max_pool_2x2('pool1', h_conv1_2)
            h_conv2_1 = utils.conv_layer('conv2_1', [3, 3, 64, 128], [1, 1, 1, 1], True, False, False, "SAME", h_pool1)
            h_conv2_2 = utils.conv_layer('conv2_2', [3, 3, 128, 128], [1, 1, 1, 1], True, False, False, "SAME", h_conv2_1)
            h_pool2 = utils.max_pool_2x2('pool2', h_conv2_2)
            h_conv3_1 = utils.conv_layer('conv3_1', [3, 3, 128, 256], [1, 1, 1, 1], True, False, False, "SAME", h_pool2)
            h_conv3_2 = utils.conv_layer('conv3_2', [3, 3, 256, 256], [1, 1, 1, 1], True, False, False, "SAME", h_conv3_1)
            h_conv3_3 = utils.conv_layer('conv3_3', [3, 3, 256, 256], [1, 1, 1, 1], True, False, False, "SAME", h_conv3_2)
            h_pool3 = utils.max_pool_2x2('pool3', h_conv3_3)
            h_conv4_1 = utils.conv_layer('conv4_1', [3, 3, 256, 512], [1, 1, 1, 1], True, False, False, "SAME", h_pool3)
            h_conv4_2 = utils.conv_layer('conv4_2', [3, 3, 512, 512], [1, 1, 1, 1], True, False, False, "SAME", h_conv4_1)
            h_conv4_3 = utils.conv_layer('conv4_3', [3, 3, 512, 512], [1, 1, 1, 1], True, False, False, "SAME", h_conv4_2)
            h_pool4 = utils.max_pool_2x2('pool4', h_conv4_3)
            h_conv5_1 = utils.conv_layer('conv5_1', [3, 3, 512, 512], [1, 1, 1, 1], True, False, False, "SAME", h_pool4)
            h_conv5_2 = utils.conv_layer('conv5_2', [3, 3, 512, 512], [1, 1, 1, 1], True, False, False, "SAME", h_conv5_1)
            h_conv5_3 = utils.conv_layer('conv5_3', [3, 3, 512, 512], [1, 1, 1, 1], True, False, False, "SAME", h_conv5_2)
            h_pool5 = utils.max_pool_2x2('pool5', h_conv5_3)
        with tf.variable_scope('decoder', reuse=self.reuse_variables):
            h_fc6 = utils.conv_layer('fc6', [7, 7, 512, 4096], [1, 1, 1, 1], True, False, True, "VALID", h_pool5, self.params.keep_prob)
            h_fc7 = utils.conv_layer('fc7', [1, 1, 4096, 4096], [1, 1, 1, 1], True, False, True, "VALID", h_fc6, self.params.keep_prob)
            h_score_fr = utils.conv_layer('score_fr', [1, 1, 4096, self.params.num_labels], [1, 1, 1, 1], False, False, False, "VALID", h_fc7)
            # h_upscore - upscore_shape = (m, 384, 1248, num_labels)
            self.deconv = utils.deconv_layer('deconv', self.params.num_labels, 64, 32, h_score_fr)
        with tf.variable_scope('output', reuse=self.reuse_variables):
            # cropping the output image to original size
            # https://github.com/shelhamer/fcn.berkeleyvision.org/edit/master/README.md#L72
            # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/voc-fcn32s/net.py#L62
            # offset_height = (tf.shape(h_upscore)[1] - c_image_height) // 2
            # offset_width = (tf.shape(h_upscore)[2] - c_image_width) // 2
            # h_crop        = tf.image.crop_to_bounding_box(h_upscore, offset_height, offset_width, c_image_height, c_image_width)
            self.h_crop = utils.crop_tensor(self.deconv, self.params.image_height, self.params.image_width)
            self.h_argmax = tf.math.argmax(self.h_crop, axis=3)
            self.output = tf.py_func(self.batch_id_to_rgb, [self.h_argmax], tf.float32)

    def __build_loss(self):
        with tf.variable_scope('loss', reuse=self.reuse_variables):
            # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
            # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/voc-fcn32s/net.py#L63
            logits = tf.reshape(self.h_crop, (-1, self.params.num_labels), name="logits")
            y_reshaped = tf.reshape(self.y_probability, (-1, self.params.num_labels))

            # Calculate distance from actual labels using cross entropy
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_reshaped[:])

            # Take mean for total loss
            self.loss = tf.reduce_mean(self.cross_entropy, name="loss")

            self.opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name="opt")
            self.goal = self.opt.minimize(self.loss, name="goal")

    def __build_summaries(self):
        with tf.device('/cpu:0'):
            tf.summary.scalar('train_loss', self.loss, collections=["train_collection"])

            tf.summary.scalar('val_loss', self.loss, collections=["val_collection"])
            tf.summary.image('x', self.x, max_outputs=4, collections=["val_collection"])
            tf.summary.image('y', self.y, max_outputs=4, collections=["val_collection"])
            tf.summary.image('p', self.output, max_outputs=4, collections=["val_collection"])

            tf.summary.image('predicted', self.output, max_outputs=4, collections=["test_collection"])

    def rgb_to_probability(self, img):
        probability = np.zeros([self.params.image_height, self.params.image_width, self.params.num_labels])
        for label in self.params.labels:
            coords = np.where(np.all(img == np.array(label.color), axis=2))
            one_hot = np.zeros(self.params.num_labels)
            one_hot[label.id] = 1
            probability[coords[0], coords[1], :] = one_hot
        return probability

    def batch_rgb_to_probability(self, rgb_batch):
        m = rgb_batch.shape[0]
        probability_batch = np.zeros([m, self.params.image_height, self.params.image_width, self.params.num_labels])
        for i in range(m):
            probability_batch[i] = self.rgb_to_probability(rgb_batch[i])
        return probability_batch.astype(dtype=np.float32)

    def batch_id_to_rgb(self, id_batch):
        res = self.id_to_rgb[id_batch]
        return res.astype(dtype=np.float32)
