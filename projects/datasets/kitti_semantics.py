#!/usr/bin/python
from __future__ import print_function, absolute_import, division
from collections import namedtuple

import os
import glob
import scipy
import numpy as np
import scipy.misc
import vapk as utils

import tensorflow as tf
# --------------------------------------------------------------------------------
# Definitions
# --------------------------------------------------------------------------------

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.

    'color'       , # The color of this label
    ] )


# --------------------------------------------------------------------------------
# A list of all labels
# --------------------------------------------------------------------------------
labels = [
    #       name                     id   color
    Label(  'unlabeled'            , 0  , (  0,  0,  0) ),
    Label(  'dynamic'              , 1  , (111, 74,  0) ),
    Label(  'ground'               , 2  , ( 81,  0, 81) ),
    Label(  'road'                 , 3  , (128, 64,128) ),
    Label(  'sidewalk'             , 4  , (244, 35,232) ),
    Label(  'parking'              , 5  , (250,170,160) ),
    Label(  'rail track'           , 6  , (230,150,140) ),
    Label(  'building'             , 7  , ( 70, 70, 70) ),
    Label(  'wall'                 , 8  , (102,102,156) ),
    Label(  'fence'                , 9  , (190,153,153) ),
    Label(  'guard rail'           , 10 , (180,165,180) ),
    Label(  'bridge'               , 11 , (150,100,100) ),
    Label(  'tunnel'               , 12 , (150,120, 90) ),
    Label(  'pole'                 , 13 , (153,153,153) ),
    Label(  'traffic light'        , 14 , (250,170, 30) ),
    Label(  'traffic sign'         , 15 , (220,220,  0) ),
    Label(  'vegetation'           , 16 , (107,142, 35) ),
    Label(  'terrain'              , 17 , (152,251,152) ),
    Label(  'sky'                  , 18 , ( 70,130,180) ),
    Label(  'person'               , 19 , (220, 20, 60) ),
    Label(  'rider'                , 20 , (255,  0,  0) ),
    Label(  'car'                  , 21 , (  0,  0,142) ),
    Label(  'truck'                , 22 , (  0,  0, 70) ),
    Label(  'bus'                  , 23 , (  0, 60,100) ),
    Label(  'caravan'              , 24 , (  0,  0, 90) ),
    Label(  'trailer'              , 25 , (  0,  0,110) ),
    Label(  'train'                , 26 , (  0, 80,100) ),
    Label(  'motorcycle'           , 27 , (  0,  0,230) ),
    Label(  'bicycle'              , 28 , (119, 11, 32) ),
]

# --------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
# --------------------------------------------------------------------------------
name_to_label        = { label.name    : label for label in labels           }
color_to_label       = { label.color    : label for label in labels           }


# def rgb_to_probability(img, labels):
#     probability = np.zeros([img.shape[0], img.shape[1], len(labels)])
#     for label in labels:
#         coords = np.where(np.all(img == np.array(label.color), axis=2))
#         one_hot = np.zeros(len(labels))
#         one_hot[label.id] = 1
#         probability[coords[0], coords[1], :] = one_hot
#     return probability
#
#
# def convert_rgb_to_probability(rgb_batch, image_height, image_width, labels):
#     m = rgb_batch.shape[0]
#     probability_batch = np.zeros([m, image_height, image_width, len(labels)])
#     for i in range(m):
#         probability_batch[i] = rgb_to_probability(rgb_batch[i], labels)
#     return probability_batch


class KittiSemantics(object):
    def __init__(self, instances_path, ground_truth_path):
        # labels
        self.labels = labels
        self.num_labels = len(labels)

        # paths
        self.instances_path = instances_path
        self.ground_truth_path = ground_truth_path

        # data
        self.instances, self.gt = self.__initialize()
        self.num_instances = len(self.instances)

    def __initialize(self):
        """Collects the list of (instance, gt) filename's pairs"""
        instances_files = glob.glob(os.path.join(self.instances_path, '*.png'))
        instances = []
        ground_truth = []
        for instance_file in instances_files:
            filename = os.path.basename(instance_file)
            ground_truth_file = self.ground_truth_path + "/" + filename
            if not os.path.exists(ground_truth_file):
                exit("Missing label for instance {}".format(instance_file))
            instances.append(instance_file)
            ground_truth.append(ground_truth_file)

        return instances, ground_truth


if __name__ == "__main__":
    train_set = KittiSemantics(
        instances_path="/mnt/datasets/kitti/data_semantics/training/image_2",
        ground_truth_path="/mnt/datasets/kitti/data_semantics/training/semantic_rgb"
    )

    # with tf.Session() as sess:
    #     init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    #     sess.run(init_op)
    #
    #     # Initialize the iterator
    #     sess.run(iterator_init_op)
    #
    #     print(sess.run(iterator.get_next()))
    #     print(sess.run(iterator.get_next()))
    #
    #     # Move the iterator back to the beginning
    #     sess.run(init_op)
    #     print(sess.run(iterator.get_next()))

    # while True:
    #     try:
    #         sess.run(result)
    #     except tf.errors.OutOfRangeError:
    #         break
