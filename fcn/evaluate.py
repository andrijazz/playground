#!/usr/bin/env python

"""
References

* https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

"""

import tensorflow as tf
import utils
import argparse
import metrics
import os
from common import *


__author__ = "andrijazz"
__email__ = "andrija.m.djurisic@gmail.com"


def forward(samples):
    h_crop = utils.crop_tensor(upscore, c_image_height, c_image_width)
    crop_out = sess.run(h_crop, feed_dict={x: samples})
    batch_idx = np.argmax(crop_out, axis=3)
    batch_semantic = classes[batch_idx]
    return batch_idx, batch_semantic


# ---------------------------------------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------------------------------------
logger = utils.setup_logger("fcneval")
logger.info('Things are good!')

seed = 5
np.random.seed(seed)
tf.set_random_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help='Size of a batch. Default is 5.', type=int, default=5)
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
test_data_path = "../data/" + config.split + "/testing"
results_path = test_data_path + "/results"
images_path = test_data_path + "/image_2"
gt_images_path = test_data_path + '/semantic_rgb'

# load data
test_file_list = utils.load_data(images_path, gt_images_path)


# --------------------------------------------------------------------------------------------------------------------------------
# Restore model
# --------------------------------------------------------------------------------------------------------------------------------
session_config = tf.ConfigProto()
sess = tf.Session(config=session_config)
saver = tf.train.import_meta_graph('./fcn.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Now, let's access and create placeholders variables
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("input/x:0")
upscore = graph.get_tensor_by_name("output/output:0")

# ----------------------------------------------------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------------------------------------------------
batch_start = 0
per_pixel_acc = []
jaccard = []
i = 0
while batch_start < len(test_file_list):
    # get next mini-batch from training set
    batch_end = min(batch_start + config.batch_size, len(test_file_list))
    test_file_batch = test_file_list[batch_start : batch_end]
    m = len(test_file_batch)
    batch_start += config.batch_size

    # load batch into tensors
    x_batch, y_batch_idx = utils.load_samples_idx(m, test_file_batch, c_image_height, c_image_width, idx_classes)
    p_batch_idx, p_batch_semantic = forward(x_batch)

    # per pixel acc
    pp_acc_batch = metrics.per_pixel_acc(p_batch_idx, y_batch_idx)
    per_pixel_acc.extend(pp_acc_batch)

    logger.info("PPA = {}".format(pp_acc_batch))

    # jaccard
    jaccard_batch = metrics.jaccard(p_batch_idx, y_batch_idx)
    jaccard.extend(jaccard_batch)

    # save results
    utils.save_images(p_batch_semantic, test_file_batch, results_path)

np.save(results_path + "/per_pixel_acc.npy", per_pixel_acc)
np.save(results_path + "/jaccard.npy", jaccard)
logger.info("Average per pixel accuracy = {}".format(np.sum(np.array(per_pixel_acc))/len(per_pixel_acc)))
