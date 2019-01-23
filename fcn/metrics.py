from common import *


def per_pixel_acc(p_batch_idx, y_batch_idx):
    m = p_batch_idx.shape[0]
    total = c_image_height * c_image_width
    result = []
    for i in range(m):
        diff = p_batch_idx[i] - y_batch_idx[i]
        correct = np.sum(diff == 0)
        result.append(correct / total)
    return result


def iou(p_batch_idx, y_batch_idx):
    m = p_batch_idx.shape[0]
    result = []
    for i in range(m):
        jacc = np.zeros(num_classes)
        for cidx in range(num_classes):
            p_coords = np.where(p_batch_idx[i] == cidx)
            gt_coords = np.where(y_batch_idx[i] == cidx)

            p_coords_reshaped = np.concatenate(
                (p_coords[0].reshape(p_coords[0].shape[0], 1), p_coords[1].reshape(p_coords[1].shape[0], 1)),
                axis=1
            )

            gt_coords_reshaped = np.concatenate(
                (gt_coords[0].reshape(gt_coords[0].shape[0], 1), gt_coords[1].reshape(gt_coords[1].shape[0], 1)),
                axis=1
            )

            intersection = multidim_intersect(p_coords_reshaped, gt_coords_reshaped)
            union = multidim_union(p_coords_reshaped, gt_coords_reshaped)

            if union.shape[0] == 0:
                jacc[cidx] = 0
            else:
                jacc[cidx] = intersection.shape[0] / union.shape[0]

        result.append(jacc)
    return result


def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])


def multidim_union(arr1, arr2):
    arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
    union = np.union1d(arr1_view, arr2_view)
    return union.view(arr1.dtype).reshape(-1, arr1.shape[1])