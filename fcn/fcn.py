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
import os

seed = 5
np.random.seed(seed)
tf.set_random_seed(seed)

__author__ = "andrijazz"
__email__ = "andrija.m.djurisic@gmail.com"


tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.info('Things are good!')

# CPU = 1, GPU = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------------------------------------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--preprocessing', help='Preprocessing. Default is false.', type=bool, default=False)
parser.add_argument('--epochs', help='Number of training epochs. Default is 200.', type=int, default=10)
parser.add_argument('--model', help="""3 models are supported at the moment fcn32, fcn16 and fcn8. Default is fcn32.""", default="fcn32")
parser.add_argument('--batch_size', help='Size of a batch. Default is 10.', type=int, default=5)
parser.add_argument('--learning_rate', help='Learning rate parameter. Default is 0.001.', type=float, default=float(0.001))

config = parser.parse_args()
print("Configuration = {}".format(config))


# ---------------------------------------------------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------------------------------------------------

train_data_path = '../data/data_semantics/training'
images_path = train_data_path + "/image_2"
gt_images_path = train_data_path + '/semantic_rgb'

# load data
train_file_list = utils.load_data(images_path, gt_images_path)

# ---------------------------------------------------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------------------------------------------------
c_image_height = 370
c_image_width = 1224
classes = np.array(
    [[  0,   0,   0],
       [  0,   0,  70],
       [  0,   0,  90],
       [  0,   0, 110],
       [  0,   0, 142],
       [  0,   0, 230],
       [  0,  60, 100],
       [  0,  80, 100],
       [ 70,  70,  70],
       [ 70, 130, 180],
       [ 81,   0,  81],
       [102, 102, 156],
       [107, 142,  35],
       [111,  74,   0],
       [119,  11,  32],
       [128,  64, 128],
       [150, 100, 100],
       [150, 120,  90],
       [152, 251, 152],
       [153, 153, 153],
       [180, 165, 180],
       [190, 153, 153],
       [220,  20,  60],
       [220, 220,   0],
       [230, 150, 140],
       [244,  35, 232],
       [250, 170,  30],
       [250, 170, 160],
       [255,   0,   0]])

probability_classes = {
       tuple([  0,   0,   0]): np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([  0,   0,  70]): np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([  0,   0,  90]): np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([  0,   0, 110]): np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([  0,   0, 142]): np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([  0,   0, 230]): np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([  0,  60, 100]): np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([  0,  80, 100]): np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([ 70,  70,  70]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([ 70, 130, 180]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([ 81,   0,  81]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([102, 102, 156]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([107, 142,  35]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([111,  74,   0]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([119,  11,  32]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([128,  64, 128]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([150, 100, 100]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([150, 120,  90]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([152, 251, 152]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([153, 153, 153]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([180, 165, 180]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([190, 153, 153]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
       tuple([220,  20,  60]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
       tuple([220, 220,   0]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
       tuple([230, 150, 140]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
       tuple([244,  35, 232]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
       tuple([250, 170,  30]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
       tuple([250, 170, 160]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
       tuple([255,   0,   0]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
}


if config.preprocessing:
    c_image_height, c_image_width = utils.calculate_min_image_size(train_file_list)
    utils.adjust_images(train_file_list, c_image_height, c_image_width)
    exit("Preprocessing completed. Please re-run the script with preprocessing=false")

num_classes = len(classes)      # 29

# --------------------------------------------------------------------------------------------------------------------------------
# Network definition
# --------------------------------------------------------------------------------------------------------------------------------

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

session_config = tf.ConfigProto()
#session_config.gpu_options.allow_growth = True
#session_config.gpu_options.per_process_gpu_memory_fraction = 0.85
# session_config.log_device_placement = True
sess = tf.Session(config=session_config)

with tf.variable_scope('input'):
    # placeholder for two RGB WxH images
    x = tf.placeholder(tf.float32, shape=[None, c_image_height, c_image_width, 3])     # [batch, in_height, in_width, in_channels]

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
h_upscore     = utils.deconv_layer( 'upscore', num_classes, 64, 32, h_score_fr)     # upscore_shape = (m, 384, 1248, num_classes)

# cropping the output image to original size
# https://github.com/shelhamer/fcn.berkeleyvision.org/edit/master/README.md#L72
# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/voc-fcn32s/net.py#L62
offset_height = (tf.shape(h_upscore)[1] - c_image_height) // 2
offset_width = (tf.shape(h_upscore)[2] - c_image_width) // 2
h_crop        = tf.image.crop_to_bounding_box(h_upscore, offset_height, offset_width, c_image_height, c_image_width)

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
step = 0

# perform training
while epoch < config.epochs:

    # get next mini-batch from training set
    batch_end = min(batch_start + config.batch_size, len(train_file_list))
    train_file_batch = train_file_list[batch_start : batch_end]
    m = len(train_file_batch)
    batch_start += config.batch_size

    # load batch into tensors
    x_train_batch, y_train_batch = utils.load_samples(m, train_file_batch, c_image_height, c_image_width, num_classes, probability_classes)
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
    print("Step: {}, Loss: {}".format(step, l))

# --------------------------------------------------------------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------------------------------------------------------------

# test image
img_x = scipy.misc.imread('../data/data_semantics/training/image_2/000000_10.png')
test_x = np.empty([1, c_image_height, c_image_width, 3])
test_x[0, :, :, :] = img_x

img_y = scipy.misc.imread('../data/data_semantics/training/semantic_rgb/000000_10.png')

n_out = sess.run(h_crop, feed_dict={x: test_x})

# prediction - argmax returns class index for each pixel
out = np.argmax(n_out, axis=3)
result = classes[out[0, :, :]]

# plot
plt.imshow(img_x)
plt.show()

plt.imshow(img_y)
plt.show()

scipy.misc.imsave('../data/data_semantics/000000_10_result.png', result)
plt.imshow(result)
plt.show()
