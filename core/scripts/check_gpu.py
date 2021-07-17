import torch


def check_tf_2x():
    import tensorflow as tf
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
        print(c)


def check_tf_1x():
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    with tf.device('/gpu:0'):  # 1.x
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
    with tf.Session() as sess:
        print(sess.run(c))


def check_torch():
    print(torch.cuda.is_available())


check_tf_2x()
check_tf_1x()
check_torch()
