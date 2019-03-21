import tensorflow as tf


def deconv_layer(name, in_ch, out_ch, kernel_size, stride, relu, elu, batch_norm, x):
    strides = [1, stride, stride, 1]
    filter_shape = [kernel_size, kernel_size, out_ch, in_ch]    # [4, 4, 512, 1024]
    with tf.variable_scope(name):
        W = weight_variable(filter_shape, 'weights')
        b = bias_variable([out_ch], 'biases')

        in_shape = tf.shape(x)
        h = in_shape[1] * stride
        w = in_shape[2] * stride
        new_shape = tf.stack([in_shape[0], h, w, out_ch])
        h_deconv = tf.nn.conv2d_transpose(x, W, new_shape, strides=strides, padding='SAME', name=name) + b

        if batch_norm:
            h_deconv = tf.contrib.layers.batch_norm(h_deconv)

        if relu:
            h_deconv = tf.nn.relu(h_deconv)

        if elu:
            h_deconv = tf.nn.elu(h_deconv)

    return h_deconv


# bilinear tensor upsampling
def upsample(x, s):
    shape = tf.shape(x)
    # x_up = tf.image.resize_images(x, [shape[1] * s, shape[2] * s])
    x_up = tf.image.resize_nearest_neighbor(x, [shape[1] * s, shape[2] * s])
    return x_up


def crop_tensor(x, h, w):
    # cropping the output image to original size
    # https://github.com/shelhamer/fcn.berkeleyvision.org/edit/master/README.md#L72
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/voc-fcn32s/net.py#L62
    offset_height = (tf.shape(x)[1] - h) // 2
    offset_width = (tf.shape(x)[2] - w) // 2
    return tf.image.crop_to_bounding_box(x, offset_height, offset_width, h, w)


def assign_variable(name, value):
    [scope, var] = name.rsplit('/', 1)  # split on last occurrence
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        return tf.get_variable(var).assign(value)


# initialize weights from a truncated normal distribution
def weight_variable(shape, name):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name, shape, initializer=initial)


# initialize biases to constant values
def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape)
    return tf.get_variable(name, initializer=initial)


# 2d convolution with padding
# https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
def conv2d(x, W, s, padding):
    return tf.nn.conv2d(x, W, s, padding=padding)


# 2x2 max pooling with 2x2 stride and same padding
def max_pool_2x2(name, x):
    with tf.variable_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')


# convolution layer (shape = [filter_height, filter_width, in_channels, out_channels])
# TODO fix activation
def conv_layer(name, shape, stride, elu, batch_norm, dropout, padding, x, keep_prob=0.5):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W = weight_variable(shape, 'weights')
        b = bias_variable([shape[3]], 'biases')
        h_conv = conv2d(x, W, stride, padding) + b

        if batch_norm:
            h_conv = tf.contrib.layers.batch_norm(h_conv)
        if elu:
            h_conv = tf.nn.elu(h_conv)
        else:
            h_conv = 0.3 * tf.nn.sigmoid(h_conv)

        if dropout:
            h_conv = tf.nn.dropout(h_conv, keep_prob=keep_prob)

        return h_conv
