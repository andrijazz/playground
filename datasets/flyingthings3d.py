#!/usr/bin/python
from __future__ import print_function, absolute_import, division

import re
import sys
import os
import glob
import scipy
import numpy as np
import scipy.misc
import vapk as utils
import matplotlib.pyplot as plt


# read disparity data from .pfm file
def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip().decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip().decode('utf-8'))
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


# write disparity data to .pfm file
def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)


class FlyingThings3D(object):
    def __init__(self, path_img_left, path_img_right, path_gt):

        # sizes
        # self.orig_image_height = 540
        self.orig_image_height = 512
        self.orig_image_width = 960
        self.in_channels = 6        # we will pack left and right img into 3d matrix of size [h, w, 6] where left [:, :, 0:3] and right [:, :, 3:6]
        self.downsize = 1

        self.image_height = self.orig_image_height // self.downsize
        self.image_width = self.orig_image_width // self.downsize

        # paths
        self.path_img_left = path_img_left
        self.path_img_right = path_img_right

        self.path_gt = path_gt
        # pairs
        self.__initialize()
        # iters
        self.iter = utils.BatchIterator(self.file_pairs)

    def __initialize(self):
        left_images = glob.glob(os.path.join(self.path_img_left, '*.png'))
        data = []
        for left_img in left_images:
            filename = os.path.basename(left_img)

            # right
            right_img = self.path_img_right + "/" + filename
            if not os.path.exists(right_img):
                exit("Missing right image for instance {}".format(filename))

            # disparity
            gt_img_filename = filename.replace(r'.png', r'.pfm')
            gt_img = self.path_gt + "/" + gt_img_filename
            if not os.path.exists(gt_img):
                exit("Missing disparity image for instance {}".format(filename))

            data.append([left_img, right_img, gt_img])
        self.file_pairs = data

    def __load_samples(self, images):
        """Iterates through list of images and packs them into batch of size m"""
        m = len(images)
        x_batch = np.empty([m, self.image_height, self.image_width, self.in_channels])
        y_batch = np.empty([m, self.image_height, self.image_width, 1])

        for i in range(m):
            left_img_file = images[i][0]
            right_img_file = images[i][1]
            disp_img_file = images[i][2]
            left_img = scipy.misc.imread(left_img_file)
            right_img = scipy.misc.imread(right_img_file)
            x_batch[i, :, :, 0 : 3] = left_img[0 : self.image_height, 0 : self.image_width, :]
            x_batch[i, :, :, 3 : 6] = right_img[0 : self.image_height, 0 : self.image_width, :]
            disp_img, scale = readPFM(disp_img_file)
            y_batch[i, :, :, 0] = disp_img[0 : self.image_height, 0 : self.image_width]
        return x_batch, y_batch

    def load_batch(self, batch_size):
        file_batch, end_of_epoch = self.iter.next(batch_size)
        x_batch, y_batch = self.__load_samples(file_batch)
        return x_batch, y_batch, end_of_epoch


# x_left = scipy.misc.imread("/mnt/datasets/flyingthings3d/FlyingThings3D_subset_image_clean/train/image_clean/left/0000000.png")
# plt.imshow(x_left)
# plt.show()
#
# x_right = scipy.misc.imread("/mnt/datasets/flyingthings3d/FlyingThings3D_subset_image_clean/train/image_clean/right/0000000.png")
# plt.imshow(x_right)
# plt.show()
#
# y_left, scale_y_left = readPFM("/mnt/datasets/flyingthings3d/FlyingThings3D_subset_disparity/train/disparity/left/0000000.pfm")
# plt.imshow(y_left)
# plt.show()
#
# y_right, scale_y_right = readPFM("/mnt/datasets/flyingthings3d/FlyingThings3D_subset_disparity/train/disparity/right/0000000.pfm")
# plt.imshow(y_right)
# plt.show()

# dataset = FlyingThings3D(path_img_left="/mnt/datasets/flyingthings3d/FlyingThings3D_subset_image_clean/train/image_clean/left",
#                          path_img_right="/mnt/datasets/flyingthings3d/FlyingThings3D_subset_image_clean/train/image_clean/right",
#                          path_gt="/mnt/datasets/flyingthings3d/FlyingThings3D_subset_disparity/train/disparity/left")
# m = len(dataset.file_pairs)
# for i in range(5):
#     x_batch, y_batch, eoe = dataset.load_batch(3)
#     print(i)
