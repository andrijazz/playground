import tensorflow as tf


# Relative Squared Error (RSE)
# sum((p_i - gt_i) ^ 2 / (gt_avg - gt_i) ^ 2)
def rse(p, gt):
    gt_avg = tf.reduce_mean(gt)
    rse = tf.reduce_sum(tf.square(p - gt) / tf.square(gt_avg - gt))

# Relative Absolute Error (absRel) - sum(|p_i - gt_i| / |gt_avg - gt_i|)
# self.abs_rel = tf.reduce_sum(tf.abs(self.h_disp1 - self.gt) / tf.abs(gt_avg - self.gt))

# D1-all - It is the percentage of pixels for which the
# estimation error is larger than 3px and larger than 5% of the
# ground truth disparity at this pixel.

# Root Mean Squared Error (RMSE) - sqrt(1 / m * sum((p_i - gt_i) ^ 2))
# m = tf.shape(self.gt)[0]
# self.rmse = tf.sqrt(1 / m * tf.reduce_sum(tf.square(self.h_disp1 - self.gt)))

# Root Mean Squared Error Log (RMSE log) - sqrt(1 / m * sum((log p_i - log gt_i) ^ 2))
# self.rmse_log = tf.sqrt(1 / m * tf.reduce_sum(tf.square(tf.log(self.h_disp1) - tf.log(self.gt))))



