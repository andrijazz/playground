import datasets.kitti_semantics as kitti
import datasets.cityscapes as cs


def load(dataset_name, dataset_path):
    if dataset_name == "kitti":
        train_set = kitti.KittiDataset(
            path_img=dataset_path + "/kitti/data_semantics/training/image_2",
            path_gt=dataset_path + "/kitti/data_semantics/training/semantic_rgb"
        )

        val_set = kitti.KittiDataset(
            path_img=dataset_path + "/kitti/data_semantics/val/image_2",
            path_gt=dataset_path + "/kitti/data_semantics/val/semantic_rgb"
        )

        test_set = kitti.KittiDataset(
            path_img=dataset_path + "/kitti/data_semantics/testing/image_2",
            path_gt=dataset_path + "/kitti/data_semantics/testing/semantic_rgb"
        )
        return train_set, val_set, test_set

    if dataset_name == "cityscapes":
        train_set = cs.CityscapesDataset(
            path_img=dataset_path + "/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train",
            path_gt=dataset_path + "/cityscapes/gtFine_trainvaltest/gtFine/train"
        )

        val_set = cs.CityscapesDataset(
            path_img=dataset_path + "/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val",
            path_gt=dataset_path + "/cityscapes/gtFine_trainvaltest/gtFine/val"
        )

        test_set = cs.CityscapesDataset(
            path_img=dataset_path + "/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test",
            path_gt=dataset_path + "/cityscapes/gtFine_trainvaltest/gtFine/test"
        )
        return train_set, val_set, test_set

    exit("Unsupported dataset")
