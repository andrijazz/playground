import os
import tensorflow as tf
import projects.datasets.kitti as kitti


def create_train_and_val_datasets(dataset):
    if dataset == 'kitti':
        dataset_path = os.path.join(os.getenv('DATASETS'), 'data_semantics')
        images, labels = kitti.load_data(dataset_path, mode='training')
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(kitti.train_parse_function, num_parallel_calls=1)
        dataset = dataset.map(kitti.train_preprocess, num_parallel_calls=1)
        dataset = dataset.prefetch(1)
        return dataset
    else:
        exit('Unsupported dataset {}'.format(dataset))


def create_test_dataset(dataset):
    if dataset == 'KITTI':
        pass
    else:
        exit('Unsupported dataset {}'.format(dataset))

