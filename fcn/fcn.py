#!/usr/bin/env python

"""

TODO:
* data augmenation
* project structure / model classes / utils, metrics lib
* try out classification for vgg-16 on some data set
* try cityscapes data set
* instead cropping use resize (x - bilinear, y - nn)
* one-hot-encoding

batch norm (deeplearning ai exercises)

References
* https://github.com/fpanjevic/playground/tree/master/DispNet
* https://github.com/andrijazz/courses/blob/master/deeplearning/notes/deeplearning-4.ipynb
* https://github.com/shelhamer/fcn.berkeleyvision.org
* https://www.cs.toronto.edu/~frossard/post/vgg16/
* http://deeplearning.net/tutorial/fcn_2D_segm.html

Data sets
* http://www.cvlibs.net/datasets/kitti/
* https://www.cityscapes-dataset.com/


Q
* Best practices on how to organize code - loading configuration from config file?

Filip's suggestions:
https://github.com/mrharicot/monodepth
https://github.com/tinghuiz/SfMLearner

Other:
https://github.com/MrGemy95/Tensorflow-Project-Template

"""

import argparse
import tensorflow as tf
import utils
import os
import metrics
from common import *


__author__ = "andrijazz"
__email__ = "andrija.m.djurisic@gmail.com"

# ---------------------------------------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------------------------------------

# seeds
seed = 5
np.random.seed(seed)
tf.set_random_seed(seed)

# args
parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    help="""3 models are supported at the moment fcn32, fcn16 and fcn8. Default is fcn32.""",
                    type=str, default="fcn32")
parser.add_argument('--epochs', help='Number of training epochs. Default is 100.', type=int, default=100)
parser.add_argument('--batch_size', help='Size of a batch. Default is 5.', type=int, default=5)
parser.add_argument('--learning_rate', help='Learning rate parameter. Default is 0.001.', type=float, default=float(0.001))
parser.add_argument('--keep_prob', help='Dropout keep prob. Default is 0.5.', type=float, default=float(0.5))
parser.add_argument('--split', help='Split dataset. Default is split-150', type=str, default="split-150")
parser.add_argument('--gpu', help='Run on GPU. Default is False', type=bool, default=False)
parser.add_argument('--use_pretrained_model', help='Use pretrained vgg-16 weights', type=bool, default=False)
config = parser.parse_args()

# hparam_string
hparam_string = utils.make_hparam_string(config)

# out dir
OUT_DIR = LOG_DIR + "/" + hparam_string
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# logger
logger = utils.setup_logger("fcn", OUT_DIR)
logger.info('Things are good!')
logger.info("Configuration = {}".format(config))

# gpu flag
if config.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ---------------------------------------------------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------------------------------------------------
train_data_path = DATA_DIR + "/" + config.split + "/training"
images_path = train_data_path + "/image_2"
gt_images_path = train_data_path + '/semantic_rgb'

# load data
train_file_list = utils.load_data(images_path, gt_images_path)

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
h_fc6         = utils.conv_layer(   'fc6',        [7, 7, 512, 4096],       [1, 1, 1, 1],   True,  False, True, "VALID", h_pool5, config.keep_prob)
h_fc7         = utils.conv_layer(   'fc7',        [1, 1, 4096, 4096],      [1, 1, 1, 1],   True,  False, True, "VALID", h_fc6, config.keep_prob)
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
# Summaries
# ----------------------------------------------------------------------------------------------------------------------

summary_loss = tf.summary.scalar('loss', loss)

# sample used in training
train_x, train_y = utils.load_samples(1, [train_file_list[0]], c_image_height, c_image_width)
train_p = tf.placeholder(tf.float32, shape=[None, c_image_height, c_image_width, 3], name="train_p")

summary_train_x = tf.summary.image('train_x', tf.constant(train_x))
summary_train_y = tf.summary.image('train_y', tf.constant(train_y))
summary_train_p = tf.summary.image('train_p', train_p)

# test placeholders
test_y = tf.placeholder(tf.float32, shape=[None, c_image_height, c_image_width, 3], name="test_y")
test_p = tf.placeholder(tf.float32, shape=[None, c_image_height, c_image_width, 3], name="test_p")

summary_test_y = tf.summary.image('test_y', test_y)
summary_test_p = tf.summary.image('test_p', test_p)

# metric placeholder
per_pixel_accuracy = tf.placeholder(tf.float32, name="per_pixel_accuracy")
summary_per_pixel_accuracy = tf.summary.scalar('per_pixel_accuracy', per_pixel_accuracy)

test_summary_op = tf.summary.merge([summary_test_y, summary_test_p, summary_per_pixel_accuracy])

# setup tensor board
writer = tf.summary.FileWriter(OUT_DIR)
writer.add_graph(sess.graph)


# ----------------------------------------------------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------------------------------------------------

# define forward pass
def forward(samples):
    crop_out = sess.run(h_crop, feed_dict={x: samples})
    batch_idx = np.argmax(crop_out, axis=3)
    batch_semantic = classes[batch_idx]
    return batch_idx, batch_semantic


sess.run(tf.global_variables_initializer())

logger.info("Training started")

# initialized weights if configured
if config.use_pretrained_model:
    weights = np.load('vgg16_weights.npz')
    init_ops = list()
    init_ops.append(utils.assign_variable('conv1_1/weights', weights['conv1_1_W']))
    init_ops.append(utils.assign_variable('conv1_1/biases', weights['conv1_1_b']))

    init_ops.append(utils.assign_variable('conv1_2/weights', weights['conv1_2_W']))
    init_ops.append(utils.assign_variable('conv1_2/biases', weights['conv1_2_b']))

    init_ops.append(utils.assign_variable('conv2_1/weights', weights['conv2_1_W']))
    init_ops.append(utils.assign_variable('conv2_1/biases', weights['conv2_1_b']))

    init_ops.append(utils.assign_variable('conv2_2/weights', weights['conv2_2_W']))
    init_ops.append(utils.assign_variable('conv2_2/biases', weights['conv2_2_b']))

    init_ops.append(utils.assign_variable('conv3_1/weights', weights['conv3_1_W']))
    init_ops.append(utils.assign_variable('conv3_1/biases', weights['conv3_1_b']))

    init_ops.append(utils.assign_variable('conv3_2/weights', weights['conv3_2_W']))
    init_ops.append(utils.assign_variable('conv3_2/biases', weights['conv3_2_b']))

    init_ops.append(utils.assign_variable('conv3_3/weights', weights['conv3_3_W']))
    init_ops.append(utils.assign_variable('conv3_3/biases', weights['conv3_3_b']))

    init_ops.append(utils.assign_variable('conv4_1/weights', weights['conv4_1_W']))
    init_ops.append(utils.assign_variable('conv4_1/biases', weights['conv4_1_b']))

    init_ops.append(utils.assign_variable('conv4_2/weights', weights['conv4_2_W']))
    init_ops.append(utils.assign_variable('conv4_2/biases', weights['conv4_2_b']))

    init_ops.append(utils.assign_variable('conv4_3/weights', weights['conv4_3_W']))
    init_ops.append(utils.assign_variable('conv4_3/biases', weights['conv4_3_b']))

    init_ops.append(utils.assign_variable('conv5_1/weights', weights['conv5_1_W']))
    init_ops.append(utils.assign_variable('conv5_1/biases', weights['conv5_1_b']))

    init_ops.append(utils.assign_variable('conv5_2/weights', weights['conv5_2_W']))
    init_ops.append(utils.assign_variable('conv5_2/biases', weights['conv5_2_b']))

    init_ops.append(utils.assign_variable('conv5_3/weights', weights['conv5_3_W']))
    init_ops.append(utils.assign_variable('conv5_3/biases', weights['conv5_3_b']))
    sess.run(init_ops)

# randomly shuffle the training set
np.random.shuffle(train_file_list)

# initialize training
batch_start = 0
epoch = 1
step = 1

# write train sample summary
writer.add_summary(sess.run(tf.summary.merge([summary_train_x, summary_train_y])), 0)

# perform training
while epoch <= config.epochs:

    # get next mini-batch from training set
    batch_end = min(batch_start + config.batch_size, len(train_file_list))
    train_file_batch = train_file_list[batch_start : batch_end]
    m = len(train_file_batch)
    batch_start += config.batch_size

    # load batch into tensors.
    x_batch, _, y_batch_prob = utils.load_samples_prob(m, train_file_batch, c_image_height, c_image_width, probability_classes)

    # reshuffle the train set if end of epoch reached
    if batch_start >= len(train_file_list):
        np.random.shuffle(train_file_list)
        batch_start = 0
        epoch += 1

        # check how predicted test sample looks like after each epochs
        res_idx, res_semantic = forward(train_x)
        res = sess.run(summary_train_p, feed_dict={train_p: res_semantic})
        writer.add_summary(res, step)

    # run training step
    sess.run(goal, feed_dict={x: x_batch, y: y_batch_prob})

    # check loss
    if step % 10 == 0:
        res = sess.run(loss, feed_dict={x: x_batch, y: y_batch_prob})
        logger.info("step = {}, loss = {}".format(step, res))

        res = sess.run(summary_loss, feed_dict={x: x_batch, y: y_batch_prob})
        writer.add_summary(res, step)

    step += 1

# ----------------------------------------------------------------------------------------------------------------------
# Save model
# ----------------------------------------------------------------------------------------------------------------------
saver = tf.train.Saver()
saver.save(sess, OUT_DIR + "/fcn")

# ---------------------------------------------------------------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------------------------------------------------------------
test_data_path = DATA_DIR + "/" + config.split + "/testing"
images_path = test_data_path + "/image_2"
gt_images_path = test_data_path + '/semantic_rgb'

# load data
test_file_list = utils.load_data(images_path, gt_images_path)

# ---------------------------------------------------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------------------------------------------------
logger.info("Testing started")
step = 1
batch_start = 0
batch_size = 1

results = {}
while batch_start < len(test_file_list):
    # get next mini-batch from test set
    batch_end = min(batch_start + batch_size, len(test_file_list))
    test_file_batch = test_file_list[batch_start : batch_end]
    m = len(test_file_batch)
    batch_start += batch_size

    # load batch into tensors
    x_batch, y_batch, y_batch_idx = utils.load_samples_idx(m, test_file_batch, c_image_height, c_image_width, idx_classes)
    p_batch_idx, p_batch_semantic = forward(x_batch)

    # per pixel acc
    ppa_batch = metrics.per_pixel_acc(p_batch_idx, y_batch_idx)
    # iou
    iou_batch = metrics.iou(p_batch_idx, y_batch_idx)

    logger.info("step = {}, per_pixel_acc = {}".format(step, ppa_batch))

    res = sess.run(test_summary_op, feed_dict={test_y: y_batch, test_p: p_batch_semantic, per_pixel_accuracy: ppa_batch[0]})
    writer.add_summary(res, step)

    # store results
    for i in range(m):
        results[test_file_batch[i][0]] = [ppa_batch[i], iou_batch[i]]

    step += 1

np.save(OUT_DIR + "/results.npy", results)
