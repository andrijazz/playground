import os
import tensorflow as tf
import projects.utils.tf_utils as tf_utils
import projects.datasets.kitti as kitti


class FCNDataset(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        if self.dataset_name == 'kitti':
            self.idx_to_color = kitti.idx_to_color
            self.labels = kitti.labels
            self.color_to_label = kitti.color_to_label
            self.dataset_path = os.path.join(os.getenv('DATASETS'), 'data_semantics')
            self.load = kitti.load
        else:
            exit('Unsupported dataset {}'.format(self.dataset_name))

    def train_parse_function(self, filename, label):
        image_string = tf.io.read_file(filename)
        gt_image_string = tf.io.read_file(label)
        image = tf.image.decode_png(image_string, channels=3)
        gt_image = tf.image.decode_png(gt_image_string, channels=3)
        # note that tf argument ordering is (h, w)
        resized_image = tf.image.resize(image, [256, 512])
        resized_gt_image = tf.image.resize(gt_image, [256, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return resized_image, resized_gt_image

    def train_preprocess(self, image, gt_image):
        # image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        # Make sure the image is still in [0, 1]
        # image = tf.clip_by_value(image, 0.0, 1.0)
        gt_image_idx = tf_utils.rgb_to_idx_tf(gt_image, self.color_to_label)
        return image, gt_image, gt_image_idx

    def create_train_and_val_datasets(self):
        images, labels = self.load(self.dataset_path, mode='training')
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(self.train_parse_function, num_parallel_calls=1)
        dataset = dataset.map(self.train_preprocess, num_parallel_calls=1)
        dataset = dataset.prefetch(1)
        # split dataset (take 20 images for validation)
        val_dataset = dataset.take(20)
        train_dataset = dataset.skip(20)
        return train_dataset, val_dataset

    def create_test_dataset(self):
        pass

