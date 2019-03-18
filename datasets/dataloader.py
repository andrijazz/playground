import datasets.kitti_semantics as kitti
import datasets.cityscapes as cs
import datasets.flyingthings3d as flying_things
# import datasets.kitti_scene_flow as kitti_scene_flow
import datasets.kitti_raw as kitti_raw


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

    if dataset_name == "flyingthings3d":
        train_set = flying_things.FlyingThings3D(
            path_img_left=dataset_path + "/flyingthings3d/FlyingThings3D_subset_image_clean/train/image_clean/left",
            path_img_right=dataset_path + "/flyingthings3d/FlyingThings3D_subset_image_clean/train/image_clean/right",
            path_gt=dataset_path + "/flyingthings3d/FlyingThings3D_subset_disparity/train/disparity/left"
        )

        val_set = flying_things.FlyingThings3D(
            path_img_left=dataset_path + "/flyingthings3d/FlyingThings3D_subset_image_clean/val/image_clean/left",
            path_img_right=dataset_path + "/flyingthings3d/FlyingThings3D_subset_image_clean/val/image_clean/right",
            path_gt=dataset_path + "/flyingthings3d/FlyingThings3D_subset_disparity/val/disparity/left"
        )

        return train_set, val_set, None

    if dataset_name == "kitti_raw":
        train_set = kitti_raw.KittiRaw(
            data_path=dataset_path + "/kitti/",
            file_list=dataset_path + "/kitti/kitti_train_files.txt"
        )
        val_set = kitti_raw.KittiRaw(
            data_path=dataset_path + "/kitti/",
            file_list=dataset_path + "/kitti/kitti_val_files.txt"
        )
        test_set = kitti_raw.KittiRaw(
            data_path=dataset_path + "/kitti/",
            file_list=dataset_path + "/kitti/kitti_test_files.txt"
        )
        return train_set, val_set, test_set

    if dataset_name == "kitti_scene_flow":
        test_set = kitti_raw.KittiRaw(
            data_path=dataset_path + "/kitti/data_scene_flow",
            file_list=dataset_path + "/kitti/kitti_stereo_2015_test_files.txt"
        )
        return None, None, test_set

    exit("Unsupported dataset")
