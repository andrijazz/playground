import numpy as np


# Relative Squared Error (RSE)
# sum((p_i - gt_i) ^ 2 / (gt_avg - gt_i) ^ 2)
def rse(p, gt):
    m = np.shape(p)[0]
    rse = list()
    for i in range(m):
        gt_avg = np.mean(gt[i])
        rse.append(np.sum(np.square(p[i] - gt[i]) / np.square(gt_avg - gt[i])))
    return rse


# Relative Absolute Error (absRel) - sum(|p_i - gt_i| / |gt_avg - gt_i|)
# self.abs_rel = tf.reduce_sum(tf.abs(self.h_disp1 - self.gt) / tf.abs(gt_avg - self.gt))
def abs_rel(p, gt):
    m = np.shape(p)[0]
    abs_rel = list()
    for i in range(m):
        gt_avg = np.mean(gt[i])
        abs_rel.append(np.sum(np.abs(p[i] - gt[i]) / np.abs(gt_avg - gt[i])))
    return abs_rel


# D1-all - It is the percentage of pixels for which the
# estimation error is larger than 3px and larger than 5% of the
# ground truth disparity at this pixel.
# def d1_all(p, gt):
#     m = np.shape(p)[0]
#     d1_all = list()
#     for i in range(m):
#         x = np.sum(np.abs(p[i] - gt[i]) > 3)
#         y = np.abs(p[i] - gt[i]) > 0.05 * gt[i]
#         d1_all.append(np.sum(np.abs(p[i] - gt[i]) / np.abs(gt_avg - gt[i])))
#     return d1_all


# Root Mean Squared Error (RMSE) - sqrt(1 / m * sum((p_i - gt_i) ^ 2))
def rmse(p, gt):
    m = np.shape(p)[0]
    rmse = list()
    for i in range(m):
        rmse.append(np.sqrt(np.mean(np.square(p[i] - gt[i]))))
    return rmse


# Root Mean Squared Error Log (RMSE log) - sqrt(1 / m * sum((log p_i - log gt_i) ^ 2))
def rmse_log(p, gt):
    m = np.shape(p)[0]
    rmse_log = list()
    for i in range(m):
        rmse_log.append(np.sqrt(np.mean(np.square(np.log(p[i]) - np.log(gt[i])))))
    return rmse_log



