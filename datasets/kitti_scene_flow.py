#!/usr/bin/python
from __future__ import print_function, absolute_import, division

import scipy
import numpy as np
import scipy.misc
import vapk as utils
import cv2


class KittiSceneFlow(object):
    def __init__(self, lr_path, lr_files, disp_path):

        # sizes
        self.orig_image_height = 375
        self.orig_image_width = 1242

        self.image_height = 256
        self.image_width = 512
        self.num_channels = 3

        # paths
        self.lr_path = lr_path
        self.lr_files = lr_files
        self.disp_path = disp_path

        # pairs
        self.__initialize()

        # iters
        self.iter = utils.BatchIterator(self.file_pairs)

    def __initialize(self):
        self.file_pairs = list()
        with open(self.lr_files) as f:
            for line in f:
                line = line.rstrip()        # strip trailing \n
                line = line.replace(".jpg", ".png")
                images = line.rstrip().split(' ')

                filename = images[0].split('/')[-1]
                self.file_pairs.append([self.lr_path + "/" + images[0], self.lr_path + "/" + images[1], self.disp_path + "/" + filename])

    def __load_samples(self, file_batch):
        """Iterates through list of images and packs them into batch of size m"""
        m = len(file_batch)
        left_batch = np.empty([m, self.image_height, self.image_width, self.num_channels])
        right_batch = np.empty([m, self.image_height, self.image_width, self.num_channels])
        disp_batch = np.empty([m, self.orig_image_height, self.orig_image_width])

        for i in range(m):
            left_file = file_batch[i][0]
            right_file = file_batch[i][1]
            disp_file = file_batch[i][2]

            left = scipy.misc.imread(left_file)
            left = scipy.misc.imresize(left, (self.image_height, self.image_width), interp="bilinear")
            left_batch[i] = left

            right = scipy.misc.imread(right_file)
            right = scipy.misc.imresize(right, (self.image_height, self.image_width), interp="bilinear")
            right_batch[i] = right

            disp = cv2.imread(disp_file, -1)
            disp = disp.astype(np.float32) / 256
            disp = cv2.resize(disp, (self.orig_image_width, self.orig_image_height), interpolation=cv2.INTER_LINEAR)
            disp_batch[i] = disp

        return left_batch, right_batch, disp_batch

    def load_batch(self, batch_size):
        file_batch, end_of_epoch = self.iter.next(batch_size)
        left_batch, right_batch, disp_batch = self.__load_samples(file_batch)
        return left_batch, right_batch, disp_batch, end_of_epoch


# if __name__ == '__main__':
#     train_set = KittiSceneFlow(
#         path_img="/mnt/datasets/kitti/data_scene_flow/training/image_2",
#         path_gt="/mnt/datasets/kitti/data_scene_flow/training/disp_noc_0"
#     )
#
#     left_batch, right_batch, gt_left_batch, end_of_epoch = train_set.load_batch(3)
