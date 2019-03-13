from __future__ import absolute_import, division, print_function
from collections import namedtuple

import tensorflow as tf
import utils.bilinear_sampler as bilinear_sampler
import utils.tf_utils as tf_utils


monodepth_parameters = namedtuple('parameters',
                        'image_height, '
                        'image_width, '
                        'in_channels, '
                        'keep_prob, '
                        'learning_rate')

# Filip
# Kako su bas izabrali elu?
# stride 1, 2 vs paper 2, 1 https://github.com/mrharicot/monodepth/blob/5bc4bb1eaf6f6c78ec3bcda37af4eeea9fc4f0c6/monodepth_model.py#L133
# resnet encoder
# upconv deconv vs upsample (enlarge with resize nearest neighbor)


class Monodepth(object):

    def __init__(self, params, reuse_variables=None):
        self.alpha = 0.85
        self.params = params
        self.reuse_variables = reuse_variables
        self.__build_model()
        self.__build_loss()
        self.__build_metrics()
        self.__build_summaries()

    def __build_model(self):
        with tf.variable_scope('input', reuse=self.reuse_variables):
            self.left = tf.placeholder(tf.float32, shape=[None, self.params.image_height, self.params.image_width, self.params.in_channels], name="left")     # [batch, in_height, in_width, in_channels]
            self.right = tf.placeholder(tf.float32, shape=[None, self.params.image_height, self.params.image_width, self.params.in_channels], name="right")
            self.left_pyramid = self.scale_pyramid(self.left, 4)
            self.right_pyramid = self.scale_pyramid(self.right, 4)
            self.input = tf.concat(self.left, self.right, 3)
            self.gt = tf.placeholder(tf.float32, shape=[None, self.params.image_height, self.params.image_width, 2], name="gt")

        # TODO add resnet encoder
        with tf.variable_scope('encoder', reuse=self.reuse_variables): # vgg encoder
            self.h_conv1 = tf_utils.conv_layer(    name='conv1',  shape=[7, 7, 6, 32],     stride=[1, 2, 2, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=self.input)      # h / 2
            self.h_conv1b = tf_utils.conv_layer(   name='conv1b', shape=[7, 7, 32, 32],    stride=[1, 1, 1, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv1)
            self.h_conv2 = tf_utils.conv_layer(    name='conv2',  shape=[5, 5, 32, 64],    stride=[1, 2, 2, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv1b)   # h / 4
            self.h_conv2b = tf_utils.conv_layer(   name='conv2b', shape=[5, 5, 64, 64],    stride=[1, 1, 1, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv2)
            self.h_conv3 = tf_utils.conv_layer(    name='conv3',  shape=[5, 5, 64, 128],   stride=[1, 2, 2, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv2b)   # h / 8
            self.h_conv3b = tf_utils.conv_layer(   name='conv3b', shape=[3, 3, 128, 128],  stride=[1, 1, 1, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv3)
            self.h_conv4 = tf_utils.conv_layer(    name='conv4',  shape=[3, 3, 128, 256],  stride=[1, 2, 2, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv3b)   # h / 16
            self.h_conv4b = tf_utils.conv_layer(   name='conv4b', shape=[3, 3, 256, 256],  stride=[1, 1, 1, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv4)
            self.h_conv5 = tf_utils.conv_layer(    name='conv5',  shape=[3, 3, 256, 512],  stride=[1, 2, 2, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv4b)   # h / 32
            self.h_conv5b = tf_utils.conv_layer(   name='conv5b', shape=[3, 3, 512, 512],  stride=[1, 1, 1, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv5)
            self.h_conv6 = tf_utils.conv_layer(    name='conv6',  shape=[3, 3, 512, 512],  stride=[1, 2, 2, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv5b)   # h / 64
            self.h_conv6b = tf_utils.conv_layer(   name='conv6b', shape=[3, 3, 512, 512],  stride=[1, 1, 1, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv6)
            self.h_conv7 = tf_utils.conv_layer(    name='conv7',  shape=[3, 3, 512, 512],  stride=[1, 2, 2, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv6b)   # h / 128
            self.h_conv7b = tf_utils.conv_layer(   name='conv7b', shape=[3, 3, 512, 512],  stride=[1, 1, 1, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_conv7)
        with tf.variable_scope('decoder', reuse=self.reuse_variables):
            self.h_upconv7 = tf_utils.deconv_layer(name='upconv7', in_ch=512, out_ch=512, kernel_size=3, stride=2, relu=True, batch_norm=False, x=self.h_conv7b)                              # h / 64
            self.h_iconv7 = tf_utils.conv_layer(name='iconv7', shape=[3, 3, 1024, 512], stride=[1, 1, 1, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=tf.concat([self.h_upconv7, self.h_conv6b], axis=3))

            self.h_upconv6 = tf_utils.deconv_layer(name='upconv6', in_ch=512, out_ch=512, kernel_size=3, stride=2, relu=True, batch_norm=False, x=self.h_iconv7)                              # h / 32
            self.h_iconv6 = tf_utils.conv_layer(name='iconv6', shape=[3, 3, 1024, 512], stride=[1, 1, 1, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=tf.concat([self.h_upconv6, self.h_conv5b], axis=3))

            self.h_upconv5 = tf_utils.deconv_layer(name='upconv5', in_ch=512, out_ch=256, kernel_size=3, stride=2, relu=True, batch_norm=False, x=self.h_iconv6)                              # h / 16
            self.h_iconv5 = tf_utils.conv_layer(name='iconv5', shape=[3, 3, 512, 256], stride=[1, 1, 1, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=tf.concat([self.h_upconv5, self.h_conv4b], axis=3))

            self.h_upconv4 = tf_utils.deconv_layer(name='upconv4', in_ch=256, out_ch=128, kernel_size=3, stride=2, relu=True, batch_norm=False, x=self.h_iconv5)                              # h / 8
            # paper err in_ch = 128
            self.h_iconv4 = tf_utils.conv_layer(name='iconv4', shape=[3, 3, 256, 128], stride=[1, 1, 1, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=tf.concat([self.h_upconv4, self.h_conv3b], axis=3))
            self.h_disp4 = tf_utils.conv_layer(name='disp4', shape=[3, 3, 128, 2], stride=[1, 1, 1, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_iconv4)
            self.h_disp4_up = tf_utils.upsample(self.h_disp4, 2)

            self.h_upconv3 = tf_utils.deconv_layer(name='upconv3', in_ch=128, out_ch=64, kernel_size=3, stride=2, relu=True, batch_norm=False, x=self.h_iconv4)                               # h / 4
            self.h_iconv3 = tf_utils.conv_layer(name="iconv3", shape=[3, 3, 130, 64], stride=[1, 1, 1, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=tf.concat([self.h_upconv3, self.h_conv2b, self.h_disp4_up], axis=3))
            self.h_disp3 = tf_utils.conv_layer(name='disp3', shape=[3, 3, 64, 2], stride=[1, 1, 1, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_iconv3)
            self.h_disp3_up = tf_utils.upsample(self.h_disp3, 2)

            self.h_upconv2 = tf_utils.deconv_layer(name='upconv2', in_ch=64, out_ch=32, kernel_size=3, stride=2, relu=True, batch_norm=False, x=self.h_iconv3)                                # h / 2
            self.h_iconv2 = tf_utils.conv_layer(name="iconv2", shape=[3, 3, 66, 32], stride=[1, 1, 1, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=tf.concat([self.h_upconv2, self.h_conv1b, self.h_disp3_up], axis=3))
            self.h_disp2 = tf_utils.conv_layer(name='disp2', shape=[3, 3, 32, 2], stride=[1, 1, 1, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_iconv2)
            self.h_disp2_up = tf_utils.upsample(self.h_disp2, 2)

            self.h_upconv1 = tf_utils.deconv_layer(name='upconv1', in_ch=32, out_ch=16, kernel_size=3, stride=2, relu=True, batch_norm=False, x=self.h_iconv2)                                # h
            self.h_iconv1 = tf_utils.conv_layer(name="iconv1", shape=[3, 3, 18, 16], stride=[1, 1, 1, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=tf.concat([self.h_upconv1, self.h_disp2_up], axis=3))
            self.h_disp1 = tf_utils.conv_layer(name='disp1', shape=[3, 3, 16, 2], stride=[1, 1, 1, 1], elu=True, batch_norm=False, dropout=False, padding="SAME", x=self.h_iconv1)

        with tf.variable_scope('disparities'):
            self.disp_est  = [self.h_disp1, self.h_disp1, self.h_disp1, self.h_disp1]
            self.disp_left_est  = [tf.expand_dims(d[:, :, :, 0], 3) for d in self.disp_est]
            self.disp_right_est = [tf.expand_dims(d[:, :, :, 1], 3) for d in self.disp_est]

        # generate images
        with tf.variable_scope('images'):
            self.left_est  = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i])  for i in range(4)]
            self.right_est = [self.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]

        # lr consistency
        with tf.variable_scope('left-right'):
            self.right_to_left_disp = [self.generate_image_left(self.disp_right_est[i], self.disp_left_est[i])  for i in range(4)]
            self.left_to_right_disp = [self.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in range(4)]

        # disparity smoothness
        with tf.variable_scope('smoothness'):
            self.disp_left_smoothness  = self.get_disparity_smoothness(self.disp_left_est,  self.left_pyramid)
            self.disp_right_smoothness = self.get_disparity_smoothness(self.disp_right_est, self.right_pyramid)

    def __build_loss(self):
        with tf.variable_scope('loss', reuse=self.reuse_variables):
            # image reconstruction loss (appearance matching loss)
            self.l1_left = [tf.abs( self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_left  = [tf.reduce_mean(l) for l in self.l1_left]
            self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]

            self.ssim_left = [self.SSIM( self.left_est[i],  self.left_pyramid[i]) for i in range(4)]
            self.ssim_loss_left  = [tf.reduce_mean(s) for s in self.ssim_left]
            self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
            self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

            self.image_loss_right = [self.alpha * self.ssim_loss_right[i] + (1 - self.alpha) * self.l1_reconstruction_loss_right[i] for i in range(4)]
            self.image_loss_left  = [self.alpha * self.ssim_loss_left[i]  + (1 - self.alpha) * self.l1_reconstruction_loss_left[i]  for i in range(4)]
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            # disparity smoothness
            self.disp_left_loss  = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i]))  / 2 ** i for i in range(4)]
            self.disp_right_loss = [tf.reduce_mean(tf.abs(self.disp_right_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_gradient_loss = tf.add_n(self.disp_left_loss + self.disp_right_loss)

            # lr consistency
            self.lr_left_loss  = [tf.reduce_mean(tf.abs(self.right_to_left_disp[i] - self.disp_left_est[i]))  for i in range(4)]
            self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i in range(4)]
            self.lr_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)

            # total loss
            self.total_loss = self.image_loss + self.params.disp_gradient_loss_weight * self.disp_gradient_loss + self.params.lr_loss_weight * self.lr_loss
            self.opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name="opt")
            self.goal = self.opt.minimize(self.total_loss, name="goal")

    def __build_metrics(self):
        # D1-all - It is the percentage of pixels for which the
        # estimation error is larger than 3px and larger than 5% of the
        # ground truth disparity at this pixel.

        # Relative Squared Error (RSE) - sum((p_i - gt_i) ^ 2 / (gt_avg - gt_i) ^ 2)
        # self.rse = tf.reduce_sum(tf.square(self.h_disp1 - self.gt) / tf.square(gt_avg - self.gt))

        # Relative Absoulte Error (absRel) - sum(|p_i - gt_i| / |gt_avg - gt_i|)
        # self.abs_rel = tf.reduce_sum(tf.abs(self.h_disp1 - self.gt) / tf.abs(gt_avg - self.gt))

        # Root Mean Squared Error (RMSE) - sqrt(1 / m * sum((p_i - gt_i) ^ 2))
        m = tf.shape(self.gt)[0]
        self.rmse = tf.sqrt(1 / m * tf.reduce_sum(tf.square(self.h_disp1 - self.gt)))

        # Root Mean Squared Error Log (RMSE log) - sqrt(1 / m * sum((log p_i - log gt_i) ^ 2))
        self.rmse_log = tf.sqrt(1 / m * tf.reduce_sum(tf.square(tf.log(self.h_disp1) - tf.log(self.gt))))

    def __build_summaries(self):
        with tf.device('/cpu:0'):
            for i in range(4):
                tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_left[i] + self.ssim_loss_right[i], collections="train_collection")
                tf.summary.scalar('l1_loss_' + str(i), self.l1_reconstruction_loss_left[i] + self.l1_reconstruction_loss_right[i], collections="train_collection")
                tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_right[i], collections="train_collection")
                tf.summary.scalar('disp_gradient_loss_' + str(i), self.disp_left_loss[i] + self.disp_right_loss[i], collections="train_collection")
                tf.summary.scalar('lr_loss_' + str(i), self.lr_left_loss[i] + self.lr_right_loss[i], collections="train_collection")
                tf.summary.image('disp_left_est_' + str(i), self.disp_left_est[i], max_outputs=4, collections="train_collection")
                tf.summary.image('disp_right_est_' + str(i), self.disp_right_est[i], max_outputs=4, collections="train_collection")

                tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=4, collections="train_collection")
                tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=4, collections="train_collection")
                tf.summary.image('ssim_left_'  + str(i), self.ssim_left[i],  max_outputs=4, collections="train_collection")
                tf.summary.image('ssim_right_' + str(i), self.ssim_right[i], max_outputs=4, collections="train_collection")
                tf.summary.image('l1_left_'  + str(i), self.l1_left[i],  max_outputs=4, collections="train_collection")
                tf.summary.image('l1_right_' + str(i), self.l1_right[i], max_outputs=4, collections="train_collection")

                tf.summary.image('left',  self.left,   max_outputs=4, collections="train_collection")
                tf.summary.image('right', self.right,  max_outputs=4, collections="train_collection")

    @staticmethod
    def scale_pyramid(img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    @staticmethod
    def generate_image_left(img, disp):
        return bilinear_sampler.bilinear_sampler_1d_h(img, -disp)

    @staticmethod
    def generate_image_right(img, disp):
        return bilinear_sampler.bilinear_sampler_1d_h(img, disp)

    @staticmethod
    def SSIM(x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = tf_utils.max_pool_3x3(x)
        mu_y = tf_utils.max_pool_3x3(y)

        sigma_x = tf_utils.max_pool_3x3(x ** 2) - mu_x ** 2
        sigma_y = tf_utils.max_pool_3x3(y ** 2) - mu_y ** 2
        sigma_xy = tf_utils.max_pool_3x3(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    @staticmethod
    def get_disparity_smoothness(disp, pyramid):
        disp_gradients_x = [Monodepth.gradient_x(d) for d in disp]
        disp_gradients_y = [Monodepth.gradient_y(d) for d in disp]

        image_gradients_x = [Monodepth.gradient_x(img) for img in pyramid]
        image_gradients_y = [Monodepth.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

    @staticmethod
    def gradient_x(img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    @staticmethod
    def gradient_y(img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy
