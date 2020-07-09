from __future__ import absolute_import, division, print_function
from collections import namedtuple

import tensorflow as tf
import numpy as np
import vapk as utils

# constants
TRAIN = "train"
VAL = "val"
TEST = "test"

# model configuration
fcn_parameters = namedtuple('parameters',
                            'batch_size, '
                            'data, '
                            'image_height, '
                            'image_width, '
                            'labels,'
                            'num_labels,'
                            'keep_prob, '
                            'learning_rate, '
                            'model, '
                            'type')


def input_fn(instance_file, gt_file):
    instance_string = tf.read_file(instance_file)
    gt_string = tf.read_file(gt_file)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    instance = tf.image.decode_png(instance_string, channels=3)
    gt = tf.image.decode_png(gt_string, channels=3)

    # resize images (for instances we are using bilinear method)
    instance = tf.image.resize_images(instance, [image_height, image_width], method=tf.image.ResizeMethod.BILINEAR)
    gt = tf.image.resize_images(gt, [image_height, image_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # This will convert to float values in [0, 1]
    # image = tf.image.convert_image_dtype(image, tf.float32)
    return instance, gt


def preprocess_fn(image, label):
    # image = tf.image.random_flip_left_right(image)

    # image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    # image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


class fcn(object):
    """fcn model"""

    def __init__(self, params, reuse_variables=None):
        self.params = params

        # images height and width are global vars for model.py module
        global image_height
        global image_width

        image_height = self.params.image_height
        image_width = self.params.image_width

        self.reuse_variables = reuse_variables
        self.__build_input()
        self.__build_model()
        self.__build_loss()
        self.__build_summaries()

        # init id_to_color
        self.id_to_rgb = np.zeros((self.params.num_labels, 3))
        for label in self.params.labels:
            self.id_to_rgb[label.id] = label.color

    def __build_input(self):
        if self.params.type == TEST:
            # define model input
            with tf.variable_scope('input', reuse=self.reuse_variables):
                self.x = tf.placeholder(tf.float32, shape=[None, self.params.image_height, self.params.image_width, 3], name="x")  # [batch, in_height, in_width, in_channels]
                self.y = tf.placeholder(tf.float32, shape=[None, self.params.image_height, self.params.image_width, 3], name="y")  # [batch, in_height, in_width, in_channels]
                return
        if self.params.type == TRAIN or self.params.type == VAL:
            self.dataset = tf.data.Dataset.from_tensor_slices((self.params.data.instances, self.params.data.gt))
            self.dataset = self.dataset.shuffle(self.params.data.num_instances)
            self.dataset = self.dataset.map(input_fn, num_parallel_calls=4)
            # self.dataset = self.dataset.map(preprocess_fn, num_parallel_calls=4)
            self.dataset = self.dataset.batch(self.params.batch_size)
            self.dataset = self.dataset.prefetch(1)
            self.iterator = self.dataset.make_initializable_iterator()

            # define iterator init op
            self.iterator_init_op = self.iterator.initializer

            # define model input
            with tf.variable_scope('input', reuse=self.reuse_variables):
                self.x, self.y = self.iterator.get_next()
                return

        exit("Invalid model type")

    def __build_model(self):
        with tf.variable_scope('encoder', reuse=self.reuse_variables):
            # TODO: how gradient is calculated
            self.y_probability = tf.py_func(self.batch_rgb_to_probability, [self.y], tf.float32)

            # padding input by 100
            # https://github.com/shelhamer/fcn.berkeleyvision.org/edit/master/README.md#L72
            # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/voc-fcn32s/net.py#L28
            self.padded_x = tf.pad(self.x, [[0, 0], [100, 100], [100, 100], [0, 0]], "CONSTANT")

            self.h_conv1_1 = utils.conv_layer('conv1_1', [3, 3, 3, 64], [1, 1, 1, 1], True, False, False, "SAME",  self.padded_x)
            self.h_conv1_2 = utils.conv_layer('conv1_2', [3, 3, 64, 64], [1, 1, 1, 1], True, False, False, "SAME", self.h_conv1_1)
            self.h_pool1 = utils.max_pool_2x2('pool1', self.h_conv1_2)
            self.h_conv2_1 = utils.conv_layer('conv2_1', [3, 3, 64, 128], [1, 1, 1, 1], True, False, False, "SAME", self.h_pool1)
            self.h_conv2_2 = utils.conv_layer('conv2_2', [3, 3, 128, 128], [1, 1, 1, 1], True, False, False, "SAME", self.h_conv2_1)
            self.h_pool2 = utils.max_pool_2x2('pool2', self.h_conv2_2)
            self.h_conv3_1 = utils.conv_layer('conv3_1', [3, 3, 128, 256], [1, 1, 1, 1], True, False, False, "SAME", self.h_pool2)
            self.h_conv3_2 = utils.conv_layer('conv3_2', [3, 3, 256, 256], [1, 1, 1, 1], True, False, False, "SAME", self.h_conv3_1)
            self.h_conv3_3 = utils.conv_layer('conv3_3', [3, 3, 256, 256], [1, 1, 1, 1], True, False, False, "SAME", self.h_conv3_2)
            self.h_pool3 = utils.max_pool_2x2('pool3', self.h_conv3_3)
            self.h_conv4_1 = utils.conv_layer('conv4_1', [3, 3, 256, 512], [1, 1, 1, 1], True, False, False, "SAME", self.h_pool3)
            self.h_conv4_2 = utils.conv_layer('conv4_2', [3, 3, 512, 512], [1, 1, 1, 1], True, False, False, "SAME", self.h_conv4_1)
            self.h_conv4_3 = utils.conv_layer('conv4_3', [3, 3, 512, 512], [1, 1, 1, 1], True, False, False, "SAME", self.h_conv4_2)
            self.h_pool4 = utils.max_pool_2x2('pool4', self.h_conv4_3)
            self.h_conv5_1 = utils.conv_layer('conv5_1', [3, 3, 512, 512], [1, 1, 1, 1], True, False, False, "SAME", self.h_pool4)
            self.h_conv5_2 = utils.conv_layer('conv5_2', [3, 3, 512, 512], [1, 1, 1, 1], True, False, False, "SAME", self.h_conv5_1)
            self.h_conv5_3 = utils.conv_layer('conv5_3', [3, 3, 512, 512], [1, 1, 1, 1], True, False, False, "SAME", self.h_conv5_2)
            self.h_pool5 = utils.max_pool_2x2('pool5', self.h_conv5_3)
            self.h_fc6 = utils.conv_layer('fc6', [7, 7, 512, 4096], [1, 1, 1, 1], True, False, True, "VALID", self.h_pool5, self.params.keep_prob)
            self.h_fc7 = utils.conv_layer('fc7', [1, 1, 4096, 4096], [1, 1, 1, 1], True, False, True, "VALID", self.h_fc6, self.params.keep_prob)
            self.h_score_fr = utils.conv_layer('score_fr', [1, 1, 4096, self.params.num_labels], [1, 1, 1, 1], False, False, False, "VALID", self.h_fc7)
        with tf.variable_scope('decoder', reuse=self.reuse_variables):
            if self.params.model == "fcn32":
                self.deconv = utils.deconv_layer('deconv', self.params.num_labels, 64, 32, self.h_score_fr)
            elif self.params.model == "fcn16":
                h_upscore2 = utils.deconv_layer('upscore2', self.params.num_labels, 4, 2, self.h_score_fr)

                # pool4
                h_score_pool4 = utils.conv_layer('score_pool4', [1, 1, 512, self.params.num_labels], [1, 1, 1, 1], True, False, False, "SAME", self.h_pool4)
                h_score_pool4_cropped = utils.crop_tensor(h_score_pool4, tf.shape(h_upscore2)[1], tf.shape(h_upscore2)[2])
                h_fuse_pool4 = h_upscore2 + h_score_pool4_cropped

                # h_upscore16
                self.deconv = utils.deconv_layer('deconv', self.params.num_labels, 32, 16, h_fuse_pool4)

            elif self.params.model == "fcn8":
                h_upscore2 = utils.deconv_layer('upscore2', self.params.num_labels, 4, 2, self.h_score_fr)

                # pool4
                h_score_pool4 = utils.conv_layer('score_pool4', [1, 1, 512, self.params.num_labels], [1, 1, 1, 1], True, False, False, "SAME", self.h_pool4)
                h_score_pool4_cropped = utils.crop_tensor(h_score_pool4, tf.shape(h_upscore2)[1], tf.shape(h_upscore2)[2])
                h_fuse_pool4 = h_upscore2 + h_score_pool4_cropped
                h_upscore_pool4 = utils.deconv_layer('upscore_pool4', self.params.num_labels, 4, 2, h_fuse_pool4)

                # pool3
                h_score_pool3 = utils.conv_layer('score_pool3', [1, 1, 256, self.params.num_labels], [1, 1, 1, 1], True, False, False, "SAME", self.h_pool3)
                h_score_pool3_cropped = utils.crop_tensor(h_score_pool3, tf.shape(h_upscore_pool4)[1], tf.shape(h_upscore_pool4)[2])
                h_fuse_pool3 = h_upscore_pool4 + h_score_pool3_cropped

                # h_upscore8
                self.deconv = utils.deconv_layer('deconv', self.params.num_labels, 16, 8, h_fuse_pool3)

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
            self.train_op = self.opt.minimize(self.loss, name="train_op")

    def __build_summaries(self):
        with tf.device('/cpu:0'):
            collection = self.params.type + '_collection'
            tf.summary.scalar(self.params.type + '_loss', self.loss, collections=[collection])
            tf.summary.image(self.params.type + '_x', self.x, max_outputs=1, collections=[collection])
            tf.summary.image(self.params.type + '_y', self.y, max_outputs=1, collections=[collection])
            tf.summary.image(self.params.type + '_p', self.output, max_outputs=1, collections=[collection])

            self.summary_op = tf.summary.merge_all(collection)

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

    @staticmethod
    def initialize_weights_op(weights_file):
        weights = np.load(weights_file)
        init_ops = list()
        init_ops.append(utils.assign_variable('encoder/conv1_1/weights', weights['conv1_1_W']))
        init_ops.append(utils.assign_variable('encoder/conv1_1/biases', weights['conv1_1_b']))

        init_ops.append(utils.assign_variable('encoder/conv1_2/weights', weights['conv1_2_W']))
        init_ops.append(utils.assign_variable('encoder/conv1_2/biases', weights['conv1_2_b']))

        init_ops.append(utils.assign_variable('encoder/conv2_1/weights', weights['conv2_1_W']))
        init_ops.append(utils.assign_variable('encoder/conv2_1/biases', weights['conv2_1_b']))

        init_ops.append(utils.assign_variable('encoder/conv2_2/weights', weights['conv2_2_W']))
        init_ops.append(utils.assign_variable('encoder/conv2_2/biases', weights['conv2_2_b']))

        init_ops.append(utils.assign_variable('encoder/conv3_1/weights', weights['conv3_1_W']))
        init_ops.append(utils.assign_variable('encoder/conv3_1/biases', weights['conv3_1_b']))

        init_ops.append(utils.assign_variable('encoder/conv3_2/weights', weights['conv3_2_W']))
        init_ops.append(utils.assign_variable('encoder/conv3_2/biases', weights['conv3_2_b']))

        init_ops.append(utils.assign_variable('encoder/conv3_3/weights', weights['conv3_3_W']))
        init_ops.append(utils.assign_variable('encoder/conv3_3/biases', weights['conv3_3_b']))

        init_ops.append(utils.assign_variable('encoder/conv4_1/weights', weights['conv4_1_W']))
        init_ops.append(utils.assign_variable('encoder/conv4_1/biases', weights['conv4_1_b']))

        init_ops.append(utils.assign_variable('encoder/conv4_2/weights', weights['conv4_2_W']))
        init_ops.append(utils.assign_variable('encoder/conv4_2/biases', weights['conv4_2_b']))

        init_ops.append(utils.assign_variable('encoder/conv4_3/weights', weights['conv4_3_W']))
        init_ops.append(utils.assign_variable('encoder/conv4_3/biases', weights['conv4_3_b']))

        init_ops.append(utils.assign_variable('encoder/conv5_1/weights', weights['conv5_1_W']))
        init_ops.append(utils.assign_variable('encoder/conv5_1/biases', weights['conv5_1_b']))

        init_ops.append(utils.assign_variable('encoder/conv5_2/weights', weights['conv5_2_W']))
        init_ops.append(utils.assign_variable('encoder/conv5_2/biases', weights['conv5_2_b']))

        init_ops.append(utils.assign_variable('encoder/conv5_3/weights', weights['conv5_3_W']))
        init_ops.append(utils.assign_variable('encoder/conv5_3/biases', weights['conv5_3_b']))
        return init_ops
