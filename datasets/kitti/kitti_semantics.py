#!/usr/bin/python
from __future__ import print_function, absolute_import, division
from collections import namedtuple

import os
import glob
import scipy
import numpy as np
import scipy.misc
import vapk as utils
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


class KittiDataset(object):
    def __init__(self,
                 path_img,
                 path_gt):
        # sizes
        self.image_height = 370
        self.image_width = 1224
        self.num_channels = 3

        # labels
        self.labels = labels
        self.num_labels = len(labels)
        # paths
        self.path_img = path_img
        self.path_gt = path_gt
        # pairs
        self.__initialize(path_img, path_gt)
        # iters
        self.iter = utils.BatchIterator(self.file_pairs)

    def __initialize(self, path_img, path_gt):
        images = glob.glob(os.path.join(path_img, '*.png'))
        data = []
        for image in images:
            filename = os.path.basename(image)
            gt_image = path_gt + "/" + filename
            if not os.path.exists(gt_image):
                gt_image = ""
            data.append([image, gt_image])
        self.file_pairs = data

    def __load_samples(self, images):
        """Iterates through list of images and packs them into batch of size m"""
        m = len(images)
        x_batch = np.empty([m, self.image_height, self.image_width, self.num_channels])
        y_batch = np.empty([m, self.image_height, self.image_width, self.num_channels])

        for i in range(m):
            image_file = images[i][0]
            gt_image_file = images[i][1]
            image = scipy.misc.imread(image_file)
            x_batch[i] = image
            if gt_image_file: # in test set we might not have gt image
                gt_image = scipy.misc.imread(gt_image_file)
                y_batch[i] = gt_image
        return x_batch, y_batch

    def load_batch(self, batch_size):
        file_batch, end_of_epoch = self.iter.next(batch_size)
        x_batch, y_batch = self.__load_samples(file_batch)
        return x_batch, y_batch, end_of_epoch


def main():
    kitti_train = KittiDataset(
        path_img="data_semantics/training/image_2",
        path_gt="data_semantics/training/semantic_rgb"
    )

    step = 1
    epoch = 1
    batch_size = 2
    while epoch <= 3:
        x_batch, y_batch, end_of_epoch = kitti_train.load_batch(batch_size=batch_size)
        if end_of_epoch:
            print("Epoch {} completed".format(epoch))
            epoch += 1
        step += 1

    print(step)


if __name__ == "__main__":
    main()
