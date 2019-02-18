import sys

# add dataset dir to path
from settings import *
sys.path.insert(0, DATASET_DIR)

from kitti import kitti_semantics
from cityscapes import cityscapes

def load(dataset_name):
    if dataset_name == "kitti":
        train_set = kitti_semantics.KittiDataset(
            path_img=DATASET_DIR + "/kitti/data_semantics/training/image_2",
            path_gt=DATASET_DIR + "/kitti/data_semantics/training/semantic_rgb"
        )

        val_set = kitti_semantics.KittiDataset(
            path_img=DATASET_DIR + "/kitti/data_semantics/val/image_2",
            path_gt=DATASET_DIR + "/kitti/data_semantics/val/semantic_rgb"
        )

        test_set = kitti_semantics.KittiDataset(
            path_img=DATASET_DIR + "/kitti/data_semantics/testing/image_2",
            path_gt=DATASET_DIR + "/kitti/data_semantics/testing/semantic_rgb"
        )
        return train_set, val_set, test_set

    if dataset_name == "cityscapes":
        train_set = cityscapes.CityscapesDataset(
            path_img=DATASET_DIR + "/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train",
            path_gt=DATASET_DIR + "/cityscapes/gtFine_trainvaltest/gtFine/train"
        )

        val_set = cityscapes.CityscapesDataset(
            path_img=DATASET_DIR + "/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val",
            path_gt=DATASET_DIR + "/cityscapes/gtFine_trainvaltest/gtFine/val"
        )

        test_set = cityscapes.CityscapesDataset(
            path_img=DATASET_DIR + "/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test",
            path_gt=DATASET_DIR + "/cityscapes/gtFine_trainvaltest/gtFine/test"
        )
        return train_set, val_set, test_set

    exit("Unsupported dataset")
