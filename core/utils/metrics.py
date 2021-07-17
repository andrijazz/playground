import numpy as np


def rse(p, gt):
    """
    Relative Squared Error (RSE)
    sum((p_i - gt_i) ^ 2 / (gt_avg - gt_i) ^ 2)
    """
    m = np.shape(p)[0]
    rse = list()
    for i in range(m):
        rse.append(np.sum(np.square(p[i] - gt[i]) / gt[i]))
    return rse


def abs_rel(p, gt):
    """
    Relative Absolute Error (absRel) - sum(|p_i - gt_i| / |gt_avg - gt_i|)
    self.abs_rel = tf.reduce_sum(tf.abs(self.h_disp1 - self.gt) / tf.abs(gt_avg - self.gt))
    """
    m = np.shape(p)[0]
    abs_rel = list()
    for i in range(m):
        abs_rel.append(np.sum(np.abs(p[i] - gt[i]) / gt[i]))
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


def rmse(p, gt):
    """
    Root Mean Squared Error (RMSE) - sqrt(1 / m * sum((p_i - gt_i) ^ 2))
    """
    m = np.shape(p)[0]
    rmse = list()
    for i in range(m):
        rmse.append(np.sqrt(np.mean(np.square(p[i] - gt[i]))))
    return rmse


def rmse_log(p, gt):
    """
    Root Mean Squared Error Log (RMSE log) - sqrt(1 / m * sum((log p_i - log gt_i) ^ 2))
    """
    m = np.shape(p)[0]
    rmse_log = list()
    for i in range(m):
        rmse_log.append(np.sqrt(np.mean(np.square(np.log(p[i]) - np.log(gt[i])))))
    return rmse_log


def micro_iou(image, gt_image, label_values):
    """
    Compute intersection over union metric for given semantic image and
    semantic GT image.

    Args
        image           :   semantic RGB image [w, h, 3]
        gt_image        :   semantic GT RGB image [w, h, 3]
        label_values    :   RGB value per semantic class

    Returns:
        NP array of IOU values, one for each class label.
        Returns NAN IOU if there union and intersection are both zero.

    """
    iou = []
    image_arr = image.reshape(-1, 3)
    gt_image_arr = gt_image.reshape(-1, 3)

    for label_rgb in label_values:

        image_pixels = np.all(image_arr == label_rgb, axis=-1)
        gt_pixels = np.all(gt_image_arr == label_rgb, axis=-1)

        image_mask = np.zeros((image_arr.shape[0], 1), dtype=np.bool)
        image_mask[np.where(image_pixels)] = True
        gt_mask = np.zeros((image_arr.shape[0], 1), dtype=np.bool)
        gt_mask[np.where(gt_pixels)] = True

        intersection = image_mask * gt_mask
        union = image_mask + gt_mask

        if np.sum(union) > 0:
            iou.append(intersection.sum() / union.sum())
        elif np.sum(intersection) > 0:
            iou.append(0)
        else:
            iou.append(np.nan)

    return np.array(iou)
