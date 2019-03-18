#!/usr/bin/python
from __future__ import print_function, absolute_import, division

import os
import glob
import scipy
import numpy as np
import scipy.misc
import vapk as utils


class KittiSceneFlow(object):
    def __init__(self,
                 path_img,
                 path_gt):

        # sizes
        self.orig_image_height = 370
        self.orig_image_width = 1224

        self.image_height = 256
        self.image_width = 512
        self.num_channels = 3

        # paths
        self.path_img = path_img
        self.path_gt = path_gt

        # pairs
        self.__initialize()

        # iters
        self.iter = utils.BatchIterator(self.file_pairs)

    def __initialize(self):
        left_files = glob.glob(os.path.join(self.path_img, '*_10.png'))
        data = []
        for left_file in left_files:
            filename = os.path.basename(left_file)
            right_file = self.path_img + "/" + filename[:-7] + "_11.png"
            gt_left_file = self.path_gt + "/" + filename
            data.append([left_file, right_file, gt_left_file])
        self.file_pairs = data

    def __load_samples(self, file_batch):
        """Iterates through list of images and packs them into batch of size m"""
        m = len(file_batch)
        left_batch = np.empty([m, self.image_height, self.image_width, self.num_channels])
        right_batch = np.empty([m, self.image_height, self.image_width, self.num_channels])
        gt_left_batch = np.empty([m, self.image_height, self.image_width, 1])

        for i in range(m):
            left_file = file_batch[i][0]
            right_file = file_batch[i][1]
            gt_left_file = file_batch[i][2]

            left = scipy.misc.imread(left_file)
            left = scipy.misc.imresize(left, (self.image_height, self.image_width), interp="bilinear")
            left_batch[i] = left

            right = scipy.misc.imread(right_file)
            right = scipy.misc.imresize(right, (self.image_height, self.image_width), interp="bilinear")
            right_batch[i] = right

            gt_left = scipy.misc.imread(gt_left_file)
            gt_left = scipy.misc.imresize(gt_left, (self.image_height, self.image_width), interp="nearest")
            gt_left_batch[i] = gt_left.reshape((self.image_height, self.image_width, 1))

        return left_batch, right_batch, gt_left_batch

    def load_batch(self, batch_size):
        file_batch, end_of_epoch = self.iter.next(batch_size)
        left_batch, right_batch, gt_left_batch = self.__load_samples(file_batch)
        return left_batch, right_batch, gt_left_batch, end_of_epoch


# if __name__ == '__main__':
#     train_set = KittiSceneFlow(
#         path_img="/mnt/datasets/kitti/data_scene_flow/training/image_2",
#         path_gt="/mnt/datasets/kitti/data_scene_flow/training/disp_noc_0"
#     )
#
#     left_batch, right_batch, gt_left_batch, end_of_epoch = train_set.load_batch(3)
