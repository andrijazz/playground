#!/usr/bin/env python

"""

TODO:
* metrics (per pixel acc, jaccard)
* tensor board
* fcn16
* fcn8
* apply transfer learning (pretrained vgg-16)
* try out classification for vgg-16 on some data set
* try cityscapes data set

batch norm (deeplearning ai exercises)
conv layer in tf

References
* https://github.com/fpanjevic/playground/tree/master/DispNet
* https://github.com/andrijazz/courses/blob/master/deeplearning/notes/deeplearning-4.ipynb
* https://github.com/shelhamer/fcn.berkeleyvision.org
...
* https://medium.com/nanonets/how-to-do-image-segmentation-using-deep-learning-c673cc5862ef
* http://deeplearning.net/tutorial/fcn_2D_segm.html
* https://github.com/ljanyst/image-segmentation-fcn
* https://github.com/shekkizh/FCN.tensorflow/blob/master/TensorflowUtils.py

Data sets
* http://www.cvlibs.net/datasets/kitti/
* https://www.cityscapes-dataset.com/


Q
* Best practices on how to organize code - loading configuration from config file?
* Best practices on how to organize experiments?
* how to debug tf vars?

"""
import argparse
import tensorflow as tf
import utils
import os
from common import *


__author__ = "andrijazz"
__email__ = "andrija.m.djurisic@gmail.com"

# ---------------------------------------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------------------------------------
logger = utils.setup_logger("fcn")
logger.info('Things are good!')

seed = 5
np.random.seed(seed)
tf.set_random_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--preprocessing', help='Preprocessing. Default is false.', type=bool, default=False)
parser.add_argument('--epochs', help='Number of training epochs. Default is 100.', type=int, default=100)
parser.add_argument('--model',
                    help="""3 models are supported at the moment fcn32, fcn16 and fcn8. Default is fcn32.""",
                    type=str, default="fcn32")
parser.add_argument('--batch_size', help='Size of a batch. Default is 5.', type=int, default=5)
parser.add_argument('--learning_rate', help='Learning rate parameter. Default is 0.001.', type=float, default=float(0.001))
parser.add_argument('--split', help='Split dataset. Default is split-150', type=str, default="split-150")
parser.add_argument('--gpu', help='Run on GPU. Default is False', type=bool, default=False)

config = parser.parse_args()
logger.info("Configuration = {}".format(config))

if config.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ---------------------------------------------------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------------------------------------------------

train_data_path = "../data/" + config.split + "/training"
images_path = train_data_path + "/image_2"
gt_images_path = train_data_path + '/semantic_rgb'

# load data
train_file_list = utils.load_data(images_path, gt_images_path)

# ---------------------------------------------------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------------------------------------------------

if config.preprocessing:
    c_image_height, c_image_width = utils.calculate_min_image_size(train_file_list)
    utils.adjust_images(train_file_list, c_image_height, c_image_width)
    exit("Preprocessing completed. Please re-run the script with preprocessing=false")

# ----------------------------------------------------------------------------------------------------------------------
# Network definition
# ----------------------------------------------------------------------------------------------------------------------

session_config = tf.ConfigProto()
sess = tf.Session(config=session_config)

with tf.variable_scope('input'):
    # placeholder for two RGB WxH images
    x = tf.placeholder(tf.float32, shape=[None, c_image_height, c_image_width, 3], name="x")     # [batch, in_height, in_width, in_channels]

    # padding input by 100
    # https://github.com/shelhamer/fcn.berkeleyvision.org/edit/master/README.md#L72
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/voc-fcn32s/net.py#L28
    padded_x = tf.pad(x, [[0, 0], [100, 100], [100, 100], [0, 0]], "CONSTANT")
    # placeholder for output vector
    y = tf.placeholder(tf.float32, shape=[None, c_image_height, c_image_width, num_classes])     # [batch, in_height, in_width, in_channels]

# vgg-16
h_conv1_1     = utils.conv_layer(   'conv1_1',    [3, 3, 3, 64],       [1, 1, 1, 1],   True,  False, False, "SAME", padded_x)
h_conv1_2     = utils.conv_layer(   'conv1_2',    [3, 3, 64, 64],      [1, 1, 1, 1],   True,  False, False, "SAME", h_conv1_1)
h_pool1     = utils.max_pool_2x2(   'pool1',    h_conv1_2)
h_conv2_1     = utils.conv_layer(   'conv2_1',    [3, 3, 64, 128],     [1, 1, 1, 1],   True,  False, False, "SAME", h_pool1)
h_conv2_2     = utils.conv_layer(   'conv2_2',    [3, 3, 128, 128],    [1, 1, 1, 1],   True,  False, False, "SAME", h_conv2_1)
h_pool2     = utils.max_pool_2x2(   'pool2',    h_conv2_2)
h_conv3_1     = utils.conv_layer(   'conv3_1',    [3, 3, 128, 256],      [1, 1, 1, 1],   True,  False, False, "SAME", h_pool2)
h_conv3_2     = utils.conv_layer(   'conv3_2',    [3, 3, 256, 256],      [1, 1, 1, 1],   True,  False, False, "SAME", h_conv3_1)
h_conv3_3     = utils.conv_layer(   'conv3_3',    [3, 3, 256, 256],      [1, 1, 1, 1],   True,  False, False, "SAME", h_conv3_2)
h_pool3     = utils.max_pool_2x2(   'pool3',    h_conv3_3)
h_conv4_1     = utils.conv_layer(   'conv4_1',    [3, 3, 256, 512],      [1, 1, 1, 1],   True,  False, False, "SAME", h_pool3)
h_conv4_2     = utils.conv_layer(   'conv4_2',    [3, 3, 512, 512],      [1, 1, 1, 1],   True,  False, False, "SAME", h_conv4_1)
h_conv4_3     = utils.conv_layer(   'conv4_3',    [3, 3, 512, 512],      [1, 1, 1, 1],   True,  False, False, "SAME", h_conv4_2)
h_pool4     = utils.max_pool_2x2(   'pool4',    h_conv4_3)
h_conv5_1     = utils.conv_layer(   'conv5_1',    [3, 3, 512, 512],      [1, 1, 1, 1],   True,  False, False, "SAME", h_pool4)
h_conv5_2     = utils.conv_layer(   'conv5_2',    [3, 3, 512, 512],      [1, 1, 1, 1],   True,  False, False, "SAME", h_conv5_1)
h_conv5_3     = utils.conv_layer(   'conv5_3',    [3, 3, 512, 512],      [1, 1, 1, 1],   True,  False, False, "SAME", h_conv5_2)
h_pool5     = utils.max_pool_2x2(   'pool5',    h_conv5_3)

# end of vgg-16 (vgg-16 would now have 2 fully connected layers with 4096 units each and then softmax)
# ... continuing with fc layers
h_fc6         = utils.conv_layer(   'fc6',        [7, 7, 512, 4096],       [1, 1, 1, 1],   True,  False, True, "VALID", h_pool5)
h_fc7         = utils.conv_layer(   'fc7',        [1, 1, 4096, 4096],      [1, 1, 1, 1],   True,  False, True, "VALID", h_fc6)
h_score_fr    = utils.conv_layer(   'score_fr',   [1, 1, 4096, num_classes], [1, 1, 1, 1], False,  False, False, "VALID", h_fc7)

output = None
if config.model == "fcn32":
    # h_upscore - upscore_shape = (m, 384, 1248, num_classes)
    output = utils.deconv_layer('output', num_classes, 64, 32, h_score_fr)
elif config.model == "fcn16":
    # upscore_shape = (m, 384, 1248, num_classes)
    h_upscore2 = utils.deconv_layer('upscore2', num_classes, 4, 2, h_score_fr)

    # pool4
    h_score_pool4 = utils.conv_layer('score_pool4', [1, 1, 512, num_classes], [1, 1, 1, 1],   True,  False, False, "SAME", h_pool4)
    h_score_pool4_cropped = utils.crop_tensor(h_score_pool4, tf.shape(h_upscore2)[1], tf.shape(h_upscore2)[2])
    h_fuse_pool4 = h_upscore2 + h_score_pool4_cropped
    # h_upscore16
    output = utils.deconv_layer('output', num_classes, 32, 16, h_fuse_pool4)
elif config.model == "fcn8":
    # upscore_shape = (m, 384, 1248, num_classes)
    h_upscore2 = utils.deconv_layer('upscore2', num_classes, 4, 2, h_score_fr)

    # pool4
    h_score_pool4 = utils.conv_layer('score_pool4', [1, 1, 512, num_classes], [1, 1, 1, 1],   True,  False, False, "SAME", h_pool4)
    h_score_pool4_cropped = utils.crop_tensor(h_score_pool4, tf.shape(h_upscore2)[1], tf.shape(h_upscore2)[2])
    h_fuse_pool4 = h_upscore2 + h_score_pool4_cropped
    h_upscore_pool4 = utils.deconv_layer('upscore_pool4', num_classes, 4, 2, h_fuse_pool4)

    # pool3
    h_score_pool3 = utils.conv_layer('score_pool3', [1, 1, 256, num_classes], [1, 1, 1, 1],   True,  False, False, "SAME", h_pool3)
    h_score_pool3_cropped = utils.crop_tensor(h_score_pool3, tf.shape(h_upscore_pool4)[1], tf.shape(h_upscore_pool4)[2])
    h_fuse_pool3 = h_upscore_pool4 + h_score_pool3_cropped
    # h_upscore8
    output = utils.deconv_layer('output', num_classes, 16, 8, h_fuse_pool3)
else:
    exit("Unknown model!")

# cropping the output image to original size
# https://github.com/shelhamer/fcn.berkeleyvision.org/edit/master/README.md#L72
# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/voc-fcn32s/net.py#L62
# offset_height = (tf.shape(h_upscore)[1] - c_image_height) // 2
# offset_width = (tf.shape(h_upscore)[2] - c_image_width) // 2
# h_crop        = tf.image.crop_to_bounding_box(h_upscore, offset_height, offset_width, c_image_height, c_image_width)
h_crop = utils.crop_tensor(output, c_image_height, c_image_width)

with tf.variable_scope('loss'):
    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/voc-fcn32s/net.py#L63
    logits = tf.reshape(h_crop, (-1, num_classes), name="fcn_logits")
    y_reshaped = tf.reshape(y, (-1, num_classes))

    # Calculate distance from actual labels using cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_reshaped[:])

    # Take mean for total loss
    loss = tf.reduce_mean(cross_entropy, name="fcn_loss")

opt = tf.train.AdamOptimizer(learning_rate=config.learning_rate, name="fcn_opt")
goal = opt.minimize(loss, name="fcn_goal")

# ----------------------------------------------------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------------------------------------------------
sess.run(tf.global_variables_initializer())

# randomly shuffle the training set
np.random.shuffle(train_file_list)

# initialize training
batch_start = 0
epoch = 0
J = []
curr_loss = 0
step = 0

# perform training
while epoch < config.epochs:

    # get next mini-batch from training set
    batch_end = min(batch_start + config.batch_size, len(train_file_list))
    train_file_batch = train_file_list[batch_start : batch_end]
    m = len(train_file_batch)
    batch_start += config.batch_size

    # load batch into tensors
    x_batch, y_batch_prob = utils.load_samples_prob(m, train_file_batch, c_image_height, c_image_width, probability_classes)
    # reshuffle the train set if end of epoch reached
    if batch_start >= len(train_file_list):
        logger.info("Epoch: {}, Loss: {}".format(epoch, curr_loss))
        J.append(curr_loss)
        np.random.shuffle(train_file_list)
        batch_start = 0
        epoch += 1

    # run training step
    sess.run(goal, feed_dict={x: x_batch, y: y_batch_prob})
    curr_loss = sess.run(loss, feed_dict={x: x_batch, y: y_batch_prob})
    step += 1
    logger.debug("Step: {}, Loss: {}".format(step, curr_loss))


# ----------------------------------------------------------------------------------------------------------------------
# Save model
# ----------------------------------------------------------------------------------------------------------------------
saver = tf.train.Saver()
saver.save(sess, "./{}".format(config.model))
