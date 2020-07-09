import tensorflow as tf


# initialize weights from a truncated normal distribution
def weight_variable(shape, name):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name, shape, initializer=initial)


# initialize biases to constant values
def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape)
    return tf.get_variable(name, initializer=initial)


def conv2d(input, W, s, padding):
    return tf.nn.conv2d(input, W, s, padding=padding)


def max_pool(input, name, ksize, strides, padding, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        return tf.nn.max_pool(input, ksize, strides, padding)


def conv_layer(input, name, shape, stride, padding, activation=None, reuse=None, dropout=None):
    with tf.variable_scope(name, reuse=reuse):
        W = weight_variable(shape, 'weights')
        b = bias_variable([shape[3]], 'biases')
        h_conv = conv2d(input, W, stride, padding) + b
        if activation:
            h_conv = activation(h_conv)
        if dropout:
            h_conv = tf.nn.dropout(h_conv, dropout)

        return h_conv


def deconv_layer(input, name, in_ch, out_ch, kernel_size, stride, activation=None, reuse=None):
    strides = [1, stride, stride, 1]
    filter_shape = [kernel_size, kernel_size, out_ch, in_ch]    # [4, 4, 512, 1024]
    with tf.variable_scope(name, reuse=reuse):
        W = weight_variable(filter_shape, 'weights')
        b = bias_variable([out_ch], 'biases')

        in_shape = tf.shape(input)
        h = in_shape[1] * stride
        w = in_shape[2] * stride
        new_shape = tf.stack([in_shape[0], h, w, out_ch])
        h_deconv = tf.nn.conv2d_transpose(input, W, new_shape, strides=strides, padding='SAME', name=name) + b
        if activation:
            h_deconv = activation(h_deconv)

        return h_deconv
