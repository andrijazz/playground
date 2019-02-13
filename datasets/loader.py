from .kitti import kitti_semantics


def get(dataset_name):
    if dataset_name == "kitti":
        train_set = kitti_semantics.KittiDataset(
            image_height=370,
            image_width=1224,
            path_img="/home/andrijazz/source/andrijazz/playground/datasets/kitti/data_semantics/training/image_2",
            path_gt="/home/andrijazz/source/andrijazz/playground/datasets/kitti/data_semantics/training/semantic_rgb"
        )

        val_set = kitti_semantics.KittiDataset(
            image_height=370,
            image_width=1224,
            path_img="/home/andrijazz/source/andrijazz/playground/datasets/kitti/data_semantics/val/image_2",
            path_gt="/home/andrijazz/source/andrijazz/playground/datasets/kitti/data_semantics/val/semantic_rgb"
        )

        test_set = kitti_semantics.KittiDataset(
            image_height=370,
            image_width=1224,
            path_img="/home/andrijazz/source/andrijazz/playground/datasets/kitti/data_semantics/testing/image_2",
            path_gt=""
        )
        return train_set, val_set, test_set

    if dataset_name == "cityscapes":
        # dataset = cityscapes.CityscapesDataset()
        return None

    exit("Unsupported dataset")
