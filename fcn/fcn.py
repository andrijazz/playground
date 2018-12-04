#!/usr/bin/env python

"""

TODO:
* metrics (per pixel acc, jaccard)
* logger
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
* Should I change the filters sizes considering that kitty images are 1245x375 and vgg-16 was tested on image-net dataset
 (256x256)?
* Best practices on how to organize code - loading configuration from config file?
* Best practices on how to organize experiments?
* how to debug tf vars?

"""

import argparse
import tensorflow as tf
import utils
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

seed = 5
np.random.seed(seed)
tf.set_random_seed(seed)

__author__ = "andrijazz"
__email__ = "andrija.m.djurisic@gmail.com"


tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.info('Things are good!')

# ---------------------------------------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', help='Number of training epochs. Default is 200.', type=int, default=200)
parser.add_argument('--model', help="""3 models are supported at the moment fcn32, fcn16 and fcn8. Default is fcn32.""", default="fcn32")
parser.add_argument('--batch_size', help='Size of a batch. Default is 10.', type=int, default=1)
parser.add_argument('--learning_rate', help='Learning rate parameter. Default is 0.001.', type=float, default=float(0.001))

num_classes = 40
c_image_height = 375
c_image_width = 1240
c_image_downsize = 1

config = parser.parse_args()
print("Configuration = {}".format(config))

# ---------------------------------------------------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------------------------------------------------

train_data_path = '../data/data_semantics/training'
images_path = train_data_path + "/image_2"
gt_images_path = train_data_path + '/semantic_rgb'

# resize images
utils.resize_images(images_path, c_image_height, c_image_width)
utils.resize_images(gt_images_path, c_image_height, c_image_width)

# load data
train_file_list = utils.load_data(images_path, gt_images_path)

# # plot some image
# some_index = np.random.randint(images.size)
#
# img = images[some_index]
# plt.imshow(img)
# plt.show()
#
# gt_img = gt_images[some_index]
# plt.imshow(gt_img)
# plt.show()


# --------------------------------------------------------------------------------------------------------------------------------
# Network definition
# --------------------------------------------------------------------------------------------------------------------------------

sess = tf.InteractiveSession()
with tf.variable_scope('input'):
    # placeholder for two RGB WxH images
    x = tf.placeholder(tf.float32, shape=[None, None, None, 3])     # [batch, in_height, in_width, in_channels]
    # placeholder for output vector
    y = tf.placeholder(tf.float32, shape=[None, None, None, 3])     # [batch, in_height, in_width, in_channels]

# vgg-16
h_conv1_1     = utils.conv_layer(   'conv1_1',    [224, 224, 3, 64],       [1, 1, 1, 1],   True,  False, False, "SAME", x)
h_conv1_2     = utils.conv_layer(   'conv1_2',    [224, 224, 64, 64],      [1, 1, 1, 1],   True,  False, False, "SAME", h_conv1_1)
h_pool1     = utils.max_pool_2x2(   'pool1',    h_conv1_2)
h_conv2_1     = utils.conv_layer(   'conv2_1',    [112, 112, 64, 128],     [1, 1, 1, 1],   True,  False, False, "SAME", h_pool1)
h_conv2_2     = utils.conv_layer(   'conv2_2',    [112, 112, 128, 128],    [1, 1, 1, 1],   True,  False, False, "SAME", h_conv2_1)
h_pool2     = utils.max_pool_2x2(   'pool2',    h_conv2_2)
h_conv3_1     = utils.conv_layer(   'conv3_1',    [56, 56, 128, 256],      [1, 1, 1, 1],   True,  False, False, "SAME", h_pool2)
h_conv3_2     = utils.conv_layer(   'conv3_2',    [56, 56, 256, 256],      [1, 1, 1, 1],   True,  False, False, "SAME", h_conv3_1)
h_conv3_3     = utils.conv_layer(   'conv3_3',    [56, 56, 256, 256],      [1, 1, 1, 1],   True,  False, False, "SAME", h_conv3_2)
h_pool3     = utils.max_pool_2x2(   'pool3',    h_conv3_3)
h_conv4_1     = utils.conv_layer(   'conv4_1',    [28, 28, 256, 512],      [1, 1, 1, 1],   True,  False, False, "SAME", h_pool3)
h_conv4_2     = utils.conv_layer(   'conv4_2',    [28, 28, 512, 512],      [1, 1, 1, 1],   True,  False, False, "SAME", h_conv4_1)
h_conv4_3     = utils.conv_layer(   'conv4_3',    [28, 28, 512, 512],      [1, 1, 1, 1],   True,  False, False, "SAME", h_conv4_2)
h_pool4     = utils.max_pool_2x2(   'pool4',    h_conv4_3)
h_conv5_1     = utils.conv_layer(   'conv5_1',    [14, 14, 512, 512],      [1, 1, 1, 1],   True,  False, False, "SAME", h_pool4)
h_conv5_2     = utils.conv_layer(   'conv5_2',    [14, 14, 512, 512],      [1, 1, 1, 1],   True,  False, False, "SAME", h_conv5_1)
h_conv5_3     = utils.conv_layer(   'conv5_3',    [14, 14, 512, 512],      [1, 1, 1, 1],   True,  False, False, "SAME", h_conv5_2)
h_pool5     = utils.max_pool_2x2(   'pool5',    h_conv5_3)

# end of vgg-16 (vgg-16 would now have 2 fully connected layers with 4096 units each and then softmax)
# ... continuing with fc layers
h_fc6         = utils.conv_layer(   'fc6',        [7, 7, 512, 4096],       [1, 1, 1, 1],   True,  False, True, "SAME", h_pool5)
h_fc7         = utils.conv_layer(   'fc7',        [1, 1, 4096, 4096],      [1, 1, 1, 1],   True,  False, True, "SAME", h_fc6)
h_score_fr    = utils.conv_layer(   'score_fr',   [1, 1, 4096, num_classes], [1, 1, 1, 1], False,  False, False, "SAME", h_fc7)
h_upscore     = utils.deconv_layer( 'upscore', num_classes, 64, 32, h_score_fr)

# n.score = crop(n.upscore, n.data)
# n.loss = L.SoftmaxWithLoss(n.score, n.label, loss_param=dict(normalize=False, ignore_label=255))

with tf.variable_scope('loss'):
    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(h_upscore, (-1, num_classes), name="fcn_logits")
    y_reshaped = tf.reshape(y, (-1, num_classes))

    # Calculate distance from actual labels using cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped[:])

    # Take mean for total loss
    loss = tf.reduce_mean(cross_entropy, name="fcn_loss")

opt = tf.train.AdamOptimizer(learning_rate=config.learning_rate, name="fcn_opt")
goal = opt.minimize(loss, name="fcn_goal")


# --------------------------------------------------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------------------------------------------------
sess.run(tf.global_variables_initializer())

# randomly shuffle the training set
np.random.shuffle(train_file_list)

# initialize training
batch_start = 0
epoch = 0
J = []
l = 0
step = 1

# perform training
while epoch < config.epochs:

    # get next mini-batch from training set
    batch_end = min(batch_start + config.batch_size, len(train_file_list))
    train_file_batch = train_file_list[batch_start : batch_end]
    m = len(train_file_batch)
    batch_start += config.batch_size

    # load batch into tensors
    x_train_batch, y_train_batch = utils.load_samples(m, train_file_batch, c_image_height, c_image_width, c_image_downsize)

    # reshuffle the train set if end of epoch reached
    if batch_start >= len(train_file_list):
        np.random.shuffle(train_file_list)
        batch_start = 0
        epoch += 1
        print("Epoch: {}, Loss: {}".format(epoch, l))

    # run training step
    sess.run(goal, feed_dict={x: x_train_batch, y: y_train_batch})
    l = sess.run(loss, feed_dict={x: x_train_batch, y: y_train_batch})
    J.append(l)
    step += 1
    print("Epoch: {}, Loss: {}".format(epoch, l))
    # TODO compute validation image loss
    # if step % 50 == 0:
    #     image_0_loss = sess.run(loss, feed_dict={x : x_test_sample, y_ : y_test_sample, w_ : loss_weight, lr_ : learning_rate})
    #     print("Image 0 loss: %g"%(image_0_loss))

# image = scipy.misc.imread('../data/data_semantics/training/image_2/000000_10.png')
# x_batch = np.empty([1, c_image_height, c_image_width, 3])
# x_batch[0, :, :, :] = image
# r = sess.run(h_conv1_1, feed_dict={x: x_batch})
