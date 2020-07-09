#!/usr/bin/python
from __future__ import print_function, absolute_import, division

import scipy
import numpy as np
import scipy.misc
import vapk as utils


class KittiRaw(object):
    def __init__(self, data_path, file_list):

        # sizes
        self.orig_image_height = 370
        self.orig_image_width = 1224

        self.image_height = 256
        self.image_width = 512
        self.num_channels = 3

        # paths
        self.data_path = data_path
        self.file_list = file_list

        # pairs
        self.__initialize()

        # iters
        self.iter = utils.BatchIterator(self.file_pairs)

    def __initialize(self):
        self.file_pairs = list()
        with open(self.file_list) as f:
            for line in f:
                line = line.rstrip()        # strip trailing \n
                line = line.replace(".jpg", ".png")
                images = line.rstrip().split(' ')
                self.file_pairs.append([self.data_path + "/" + images[0], self.data_path + "/" + images[1]])

    def __load_samples(self, file_batch):
        """Iterates through list of images and packs them into batch of size m"""
        m = len(file_batch)
        left_batch = np.empty([m, self.image_height, self.image_width, self.num_channels])
        right_batch = np.empty([m, self.image_height, self.image_width, self.num_channels])

        for i in range(m):
            left_file = file_batch[i][0]
            right_file = file_batch[i][1]

            left = scipy.misc.imread(left_file)
            left = scipy.misc.imresize(left, (self.image_height, self.image_width), interp="bilinear")
            left_batch[i] = left

            right = scipy.misc.imread(right_file)
            right = scipy.misc.imresize(right, (self.image_height, self.image_width), interp="bilinear")
            right_batch[i] = right

        return left_batch, right_batch

    def load_batch(self, batch_size):
        file_batch, end_of_epoch = self.iter.next(batch_size)
        left_batch, right_batch = self.__load_samples(file_batch)
        return left_batch, right_batch, end_of_epoch


# if __name__ == '__main__':
#     train_set = KittiSceneFlow(
#         path_img="/mnt/datasets/kitti/data_scene_flow/training/image_2",
#         path_gt="/mnt/datasets/kitti/data_scene_flow/training/disp_noc_0"
#     )
#
#     left_batch, right_batch, gt_left_batch, end_of_epoch = train_set.load_batch(3)
