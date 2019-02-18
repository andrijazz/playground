import sys

# add dataset dir to path
from settings import *
sys.path.insert(0, DATASET_DIR)

from kitti import kitti_semantics


def load(dataset_name):
    if dataset_name == "kitti":
        train_set = kitti_semantics.KittiDataset(
            image_height=370,
            image_width=1224,
            path_img=DATASET_DIR + "/kitti/data_semantics/training/image_2",
            path_gt=DATASET_DIR + "/kitti/data_semantics/training/semantic_rgb"
        )

        val_set = kitti_semantics.KittiDataset(
            image_height=370,
            image_width=1224,
            path_img=DATASET_DIR + "/kitti/data_semantics/val/image_2",
            path_gt=DATASET_DIR + "/kitti/data_semantics/val/semantic_rgb"
        )

        test_set = kitti_semantics.KittiDataset(
            image_height=370,
            image_width=1224,
            path_img=DATASET_DIR + "/kitti/data_semantics/testing/image_2",
            path_gt=DATASET_DIR + "/kitti/data_semantics/testing/semantic_rgb"
        )
        return train_set, val_set, test_set

    if dataset_name == "cityscapes":
        # dataset = cityscapes.CityscapesDataset()
        return None

    exit("Unsupported dataset")
