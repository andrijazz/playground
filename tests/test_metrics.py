import unittest
import tensorflow as tf
import numpy as np
import utils.metrics as metrics


class TestMetrics(unittest.TestCase):

    def test_rse(self):
        with tf.Session() as sess:
            p = tf.constant(np.arange(24).reshape((2, 3, 4)))
            gt = tf.constant(np.arange(24).reshape((2, 3, 4)) * 2)

            gt_avg = tf.reduce_mean(gt, axis=1)
            print(sess.run(gt_avg))
            # p = tf.constant([[1, 2, 3], [4, 5, 6]])
            # tf_input = tf.constant(test_input, dtype=tf.float32)
            # tf_indices_d1 = tf.constant(test_indices_d1, dtype=tf.int32)
            # tf_indices_d2 = tf.constant(test_indices_d2, dtype=tf.int32)
            # tf_result = get_entry_tf(tf_input, tf_indices_d1, tf_indices_d2, batch_size)
            # tf_result = sess.run(tf_result)
            # # check that outputs are similar
            # success = success and np.allclose(test_result, tf_result)

        # metrics.rse()
        pass


if __name__ == '__main__':
    unittest.main()
