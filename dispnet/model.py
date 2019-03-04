from __future__ import absolute_import, division, print_function
from collections import namedtuple

import tensorflow as tf
import numpy as np
import vapk as utils


dispnet_parameters = namedtuple('parameters',
                        'image_height, '
                        'image_width, '
                        'in_channels, '
                        'keep_prob, '
                        'learning_rate')


# TODO move to vapk
def deconv_layer(name, in_ch, out_ch, kernel_size, stride, relu, batch_norm, x):
    strides = [1, stride, stride, 1]
    filter_shape = [kernel_size, kernel_size, out_ch, in_ch]    # [4, 4, 512, 1024]
    with tf.variable_scope(name):
        W = utils.weight_variable(filter_shape, 'weights')
        b = utils.bias_variable([out_ch], 'biases')

        in_shape = tf.shape(x)
        h = in_shape[1] * stride
        w = in_shape[2] * stride
        new_shape = tf.stack([in_shape[0], h, w, out_ch])
        h_deconv = tf.nn.conv2d_transpose(x, W, new_shape, strides=strides, padding='SAME', name=name) + b


        if batch_norm:
            h_deconv = tf.contrib.layers.batch_norm(h_deconv)

        if relu:
            h_deconv = tf.nn.relu(h_deconv)

    return h_deconv


class DispNet(object):
    """disp net model"""

    def __init__(self, params, reuse_variables=None):
        self.params = params
        self.reuse_variables = reuse_variables
        self.__build_model()
        self.__build_loss()
        self.__build_summaries()

    # TODO: Ask Filip
    def __build_model(self):
        with tf.variable_scope('input', reuse=self.reuse_variables):
            self.x = tf.placeholder(tf.float32, shape=[None, self.params.image_height, self.params.image_width, self.params.in_channels], name="x")     # [batch, in_height, in_width, in_channels]
            self.y = tf.placeholder(tf.float32, shape=[None, self.params.image_height, self.params.image_width, 1], name="y")                           # [batch, in_height, in_width, 1]

        with tf.variable_scope('contracting', reuse=self.reuse_variables):
            self.h_conv1 = utils.conv_layer (name='conv1',  shape=[7, 7, 6, 64],     stride=[1, 2, 2, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.x)
            self.h_conv2 = utils.conv_layer (name='conv2',  shape=[5, 5, 64, 128],   stride=[1, 2, 2, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv1)
            self.h_conv3a = utils.conv_layer(name='conv3a', shape=[5, 5, 128, 256],  stride=[1, 2, 2, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv2)
            self.h_conv3b = utils.conv_layer(name='conv3b', shape=[3, 3, 256, 256],  stride=[1, 1, 1, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv3a)
            self.h_conv4a = utils.conv_layer(name='conv4a', shape=[5, 5, 256, 512],  stride=[1, 2, 2, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv3b)
            self.h_conv4b = utils.conv_layer(name='conv4b', shape=[3, 3, 512, 512],  stride=[1, 1, 1, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv4a)
            self.h_conv5a = utils.conv_layer(name='conv5a', shape=[5, 5, 512, 512],  stride=[1, 2, 2, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv4b)
            self.h_conv5b = utils.conv_layer(name='conv5b', shape=[3, 3, 512, 512],  stride=[1, 1, 1, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv5a)
            self.h_conv6a = utils.conv_layer(name='conv6a', shape=[5, 5, 512, 1024], stride=[1, 2, 2, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv5b)
            self.h_conv6b = utils.conv_layer(name='conv6b', shape=[3, 3, 1024, 1024],stride=[1, 1, 1, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv6a)
        with tf.variable_scope('expanding', reuse=self.reuse_variables):
            # pr6
            self.h_pr6 = utils.conv_layer(name='pr6', shape=[3, 3, 1024, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv6b)
            self.h_upconv5 = deconv_layer(name='upconv5', in_ch=1024, out_ch=512, kernel_size=4, stride=2, relu=True, batch_norm=False, x=self.h_conv6b)
            self.h_pr6_up = deconv_layer(name='pr6_up', in_ch=1, out_ch=1, kernel_size=4, stride=2, relu=False, batch_norm=False, x=self.h_pr6)
            self.h_iconv5 = utils.conv_layer(name='iconv5', shape=[3, 3, 1025, 512], stride=[1, 1, 1, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=tf.concat([self.h_upconv5, self.h_pr6_up, self.h_conv5b], axis=3))
            # pr5
            self.h_pr5 = utils.conv_layer(name='pr5', shape=[3, 3, 512, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False, dropout=False, padding="SAME", x=self.h_iconv5)
            self.h_upconv4 = deconv_layer(name='upconv4', in_ch=512, out_ch=256, kernel_size=4, stride=2, relu=True, batch_norm=False, x=self.h_iconv5)
            self.h_pr5_up = deconv_layer(name='pr5_up', in_ch=1, out_ch=1, kernel_size=4, stride=2, relu=False, batch_norm=False, x=self.h_pr5)
            self.h_iconv4 = utils.conv_layer(name='iconv4', shape=[3, 3, 769, 256], stride=[1, 1, 1, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=tf.concat([self.h_upconv4, self.h_pr5_up, self.h_conv4b], axis=3))
            # pr4
            self.h_pr4 = utils.conv_layer(name='pr4', shape=[3, 3, 256, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False, dropout=False, padding="SAME", x=self.h_iconv4)
            self.h_upconv3 = deconv_layer(name='upconv3', in_ch=256, out_ch=128, kernel_size=4, stride=2, relu=True, batch_norm=False, x=self.h_iconv4)
            self.h_pr4_up = deconv_layer(name='pr4_up', in_ch=1, out_ch=1, kernel_size=4, stride=2, relu=False, batch_norm=False, x=self.h_pr4)
            self.h_iconv3 = utils.conv_layer(name='iconv3', shape=[3, 3, 385, 128], stride=[1, 1, 1, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=tf.concat([self.h_upconv3, self.h_pr4_up, self.h_conv3b], axis=3))
            # pr3
            self.h_pr3 = utils.conv_layer(name='pr3', shape=[3, 3, 128, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False, dropout=False, padding="SAME", x=self.h_iconv3)
            self.h_upconv2 = deconv_layer(name='upconv2', in_ch=128, out_ch=64, kernel_size=4, stride=2, relu=True, batch_norm=False, x=self.h_iconv3)
            self.h_pr3_up = deconv_layer(name='pr3_up', in_ch=1, out_ch=1, kernel_size=4, stride=2, relu=False, batch_norm=False, x=self.h_pr3)
            self.h_iconv2 = utils.conv_layer(name='iconv2', shape=[3, 3, 193, 64], stride=[1, 1, 1, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=tf.concat([self.h_upconv2, self.h_pr3_up, self.h_conv2], axis=3))
            # pr2
            self.h_pr2 = utils.conv_layer(name='pr2', shape=[3, 3, 64, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False, dropout=False, padding="SAME", x=self.h_iconv2)
            self.h_upconv1 = deconv_layer(name='upconv1', in_ch=64, out_ch=32, kernel_size=4, stride=2, relu=True, batch_norm=False, x=self.h_iconv2)
            self.h_pr2_up = deconv_layer(name='pr2_up', in_ch=1, out_ch=1, kernel_size=4, stride=2, relu=False, batch_norm=False, x=self.h_pr2)
            self.h_iconv1 = utils.conv_layer(name='iconv1', shape=[3, 3, 97, 32], stride=[1, 1, 1, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=tf.concat([self.h_upconv1, self.h_pr2_up, self.h_conv1], axis=3))
            # pr1
            self.h_pr1 = utils.conv_layer(name='pr1', shape=[3, 3, 32, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False, dropout=False, padding="SAME", x=self.h_iconv1)
            self.h_upconv0 = deconv_layer(name='upconv0', in_ch=32, out_ch=16, kernel_size=4, stride=2, relu=True, batch_norm=False, x=self.h_iconv1)
            self.h_pr1_up = deconv_layer('pr1_up', in_ch=1, out_ch=1, kernel_size=4, stride=2, relu=False, batch_norm=False, x=self.h_pr1)
            self.h_iconv0 = utils.conv_layer(name='iconv0', shape=[3, 3, 17, 16], stride=[1, 1, 1, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=tf.concat([self.h_upconv0, self.h_pr1_up], axis=3))
            # pr0
            self.h_pr0 = utils.conv_layer(name='pr0', shape=[3, 3, 16, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False, dropout=False, padding="SAME", x=self.h_iconv0)

    # def __build_model(self):
    #     with tf.variable_scope('input', reuse=self.reuse_variables):
    #         self.x = tf.placeholder(tf.float32, shape=[None, self.params.image_height, self.params.image_width, self.params.in_channels], name="x")     # [batch, in_height, in_width, in_channels]
    #         self.y = tf.placeholder(tf.float32, shape=[None, self.params.image_height, self.params.image_width, 1], name="y")                           # [batch, in_height, in_width, 1]
    #
    #     with tf.variable_scope('contracting', reuse=self.reuse_variables):
    #         self.h_conv1 = utils.conv_layer (name='conv1',  shape=[7, 7, 6, 64],     stride=[1, 2, 2, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.x)
    #         self.h_conv2 = utils.conv_layer (name='conv2',  shape=[5, 5, 64, 128],   stride=[1, 2, 2, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv1)
    #         self.h_conv3a = utils.conv_layer(name='conv3a', shape=[5, 5, 128, 256],  stride=[1, 2, 2, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv2)
    #         self.h_conv3b = utils.conv_layer(name='conv3b', shape=[3, 3, 256, 256],  stride=[1, 1, 1, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv3a)
    #         self.h_conv4a = utils.conv_layer(name='conv4a', shape=[3, 3, 256, 512],  stride=[1, 2, 2, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv3b)
    #         self.h_conv4b = utils.conv_layer(name='conv4b', shape=[3, 3, 512, 512],  stride=[1, 1, 1, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv4a)
    #         self.h_conv5a = utils.conv_layer(name='conv5a', shape=[3, 3, 512, 512],  stride=[1, 2, 2, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv4b)
    #         self.h_conv5b = utils.conv_layer(name='conv5b', shape=[3, 3, 512, 512],  stride=[1, 1, 1, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv5a)
    #         self.h_conv6a = utils.conv_layer(name='conv6a', shape=[3, 3, 512, 1024], stride=[1, 2, 2, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv5b)
    #         self.h_conv6b = utils.conv_layer(name='conv6b', shape=[3, 3, 1024, 1024],stride=[1, 1, 1, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv6a)
    #     with tf.variable_scope('expanding', reuse=self.reuse_variables):
    #         # pr6
    #         self.h_pr6 = utils.conv_layer(name='pr6', shape=[3, 3, 1024, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv6b)
    #         self.h_upconv5 = deconv_layer(name='upconv5', in_ch=1024, out_ch=512, kernel_size=4, stride=2, relu=True, batch_norm=False, x=self.h_conv6b)
    #         self.h_iconv5 = utils.conv_layer(name='iconv5', shape=[3, 3, 1025, 512], stride=[1, 1, 1, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=tf.concat([self.h_upconv5, self.h_pr6, self.h_conv5b], axis=3))
    #         # pr5
    #         self.h_pr5 = utils.conv_layer(name='pr5', shape=[3, 3, 512, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False, dropout=False, padding="SAME", x=self.h_iconv5)
    #         self.h_upconv4 = deconv_layer(name='upconv4', in_ch=512, out_ch=256, kernel_size=4, stride=2, relu=True, batch_norm=False, x=self.h_iconv5)
    #         self.h_iconv4 = utils.conv_layer(name='iconv4', shape=[3, 3, 769, 256], stride=[1, 1, 1, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=tf.concat([self.h_upconv4, self.h_pr5, self.h_conv4b], axis=3))
    #         # pr4
    #         self.h_pr4 = utils.conv_layer(name='pr4', shape=[3, 3, 256, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False, dropout=False, padding="SAME", x=self.h_iconv4)
    #         self.h_upconv3 = deconv_layer(name='upconv3', in_ch=256, out_ch=128, kernel_size=4, stride=2, relu=True, batch_norm=False, x=self.h_iconv4)
    #         self.h_iconv3 = utils.conv_layer(name='iconv3', shape=[3, 3, 385, 128], stride=[1, 1, 1, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=tf.concat([self.h_upconv3, self.h_pr4, self.h_conv3b], axis=3))
    #         # pr3
    #         self.h_pr3 = utils.conv_layer(name='pr3', shape=[3, 3, 128, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False, dropout=False, padding="SAME", x=self.h_iconv3)
    #         self.h_upconv2 = deconv_layer(name='upconv2', in_ch=128, out_ch=64, kernel_size=4, stride=2, relu=True, batch_norm=False, x=self.h_iconv3)
    #         self.h_iconv2 = utils.conv_layer(name='iconv2', shape=[3, 3, 193, 64], stride=[1, 1, 1, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=tf.concat([self.h_upconv2, self.h_pr3, self.h_conv2], axis=3))
    #         # pr2
    #         self.h_pr2 = utils.conv_layer(name='pr2', shape=[3, 3, 64, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False, dropout=False, padding="SAME", x=self.h_iconv2)
    #         self.h_upconv1 = deconv_layer(name='upconv1', in_ch=64, out_ch=32, kernel_size=4, stride=2, relu=True, batch_norm=False, x=self.h_iconv2)
    #         self.h_iconv1 = utils.conv_layer(name='iconv1', shape=[3, 3, 97, 32], stride=[1, 1, 1, 1], relu=True, batch_norm=False, dropout=False, padding="SAME", x=tf.concat([self.h_upconv1, self.h_pr2, self.h_conv1], axis=3))
    #         # pr1
    #         self.h_pr1 = utils.conv_layer(name='pr1', shape=[3, 3, 32, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False, dropout=False, padding="SAME", x=self.h_iconv1)

    def __build_loss(self):
        with tf.variable_scope('loss', reuse=self.reuse_variables):
            shape = tf.shape(self.y)

            gt_resize_1 = tf.image.resize_images(self.y, [shape[1] // 2, shape[2] // 2])
            gt_resize_2 = tf.image.resize_images(self.y, [shape[1] // 4, shape[2] // 4])
            gt_resize_3 = tf.image.resize_images(self.y, [shape[1] // 8, shape[2] // 8])
            gt_resize_4 = tf.image.resize_images(self.y, [shape[1] // 16, shape[2] // 16])
            gt_resize_5 = tf.image.resize_images(self.y, [shape[1] // 32, shape[2] // 32])

            self.loss_0 = tf.sqrt(tf.reduce_mean(tf.square(self.h_pr0 - self.y)))
            self.loss_1 = tf.sqrt(tf.reduce_mean(tf.square(self.h_pr1 - gt_resize_1)))
            self.loss_2 = tf.sqrt(tf.reduce_mean(tf.square(self.h_pr2 - gt_resize_2)))
            self.loss_3 = tf.sqrt(tf.reduce_mean(tf.square(self.h_pr3 - gt_resize_3)))
            self.loss_4 = tf.sqrt(tf.reduce_mean(tf.square(self.h_pr4 - gt_resize_4)))
            self.loss_5 = tf.sqrt(tf.reduce_mean(tf.square(self.h_pr5 - gt_resize_5)))
            self.loss = tf.add(tf.add(tf.add(tf.add(tf.add(self.loss_0, self.loss_1 / 4), self.loss_2 / 16), self.loss_3 / 64), self.loss_4 / 256), self.loss_5 / 1024, name='global_loss')

            self.opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name="opt")
            self.goal = self.opt.minimize(self.loss, name="goal")

    def __build_summaries(self):
        with tf.device('/cpu:0'):
            tf.summary.scalar('s_loss_total', self.loss, collections=["train_collection"])
            tf.summary.scalar('s_loss_0', self.loss_0, collections=["train_collection"])
            tf.summary.scalar('s_loss_1', self.loss_1, collections=["train_collection"])
            tf.summary.scalar('s_loss_2', self.loss_2, collections=["train_collection"])
            tf.summary.scalar('s_loss_3', self.loss_3, collections=["train_collection"])
            tf.summary.scalar('s_loss_4', self.loss_4, collections=["train_collection"])
            tf.summary.scalar('s_loss_5', self.loss_5, collections=["train_collection"])

            tf.summary.scalar('val_loss', self.loss, collections=["val_collection"])
            tf.summary.image('val_x', self.x, max_outputs=1, collections=["val_collection"])
            tf.summary.image('val_y', self.y, max_outputs=1, collections=["val_collection"])
            tf.summary.image('val_p', self.h_pr0, max_outputs=1, collections=["val_collection"])
            tf.summary.image('test_p', self.h_pr0, max_outputs=1, collections=["test_collection"])

