import os
from collections import namedtuple

import numpy as np
import tensorflow as tf

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.

    'color'       , # The color of this label
    ] )


# --------------------------------------------------------------------------------
# A list of all labels
# --------------------------------------------------------------------------------
labels = [
    #       name                     id   color
    Label(  'unlabeled'            , 0  , (  0,  0,  0) ),
    Label(  'dynamic'              , 1  , (111, 74,  0) ),
    Label(  'ground'               , 2  , ( 81,  0, 81) ),
    Label(  'road'                 , 3  , (128, 64,128) ),
    Label(  'sidewalk'             , 4  , (244, 35,232) ),
    Label(  'parking'              , 5  , (250,170,160) ),
    Label(  'rail track'           , 6  , (230,150,140) ),
    Label(  'building'             , 7  , ( 70, 70, 70) ),
    Label(  'wall'                 , 8  , (102,102,156) ),
    Label(  'fence'                , 9  , (190,153,153) ),
    Label(  'guard rail'           , 10 , (180,165,180) ),
    Label(  'bridge'               , 11 , (150,100,100) ),
    Label(  'tunnel'               , 12 , (150,120, 90) ),
    Label(  'pole'                 , 13 , (153,153,153) ),
    Label(  'traffic light'        , 14 , (250,170, 30) ),
    Label(  'traffic sign'         , 15 , (220,220,  0) ),
    Label(  'vegetation'           , 16 , (107,142, 35) ),
    Label(  'terrain'              , 17 , (152,251,152) ),
    Label(  'sky'                  , 18 , ( 70,130,180) ),
    Label(  'person'               , 19 , (220, 20, 60) ),
    Label(  'rider'                , 20 , (255,  0,  0) ),
    Label(  'car'                  , 21 , (  0,  0,142) ),
    Label(  'truck'                , 22 , (  0,  0, 70) ),
    Label(  'bus'                  , 23 , (  0, 60,100) ),
    Label(  'caravan'              , 24 , (  0,  0, 90) ),
    Label(  'trailer'              , 25 , (  0,  0,110) ),
    Label(  'train'                , 26 , (  0, 80,100) ),
    Label(  'motorcycle'           , 27 , (  0,  0,230) ),
    Label(  'bicycle'              , 28 , (119, 11, 32) ),
]

# --------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
# --------------------------------------------------------------------------------
name_to_label        = { label.name    : label for label in labels           }
color_to_label       = { label.color    : label for label in labels           }
idx_to_color = np.array([np.array(label.color) for label in labels])


def rgb_to_probability(img, labels):
    probability = np.zeros([img.shape[0], img.shape[1], len(labels)])
    for label in labels:
        coords = np.where(np.all(img == np.array(label.color), axis=2))
        one_hot = np.zeros(len(labels))
        one_hot[label.id] = 1
        probability[coords[0], coords[1], :] = one_hot
    return probability


def rgb_to_idx(img, labels):
    idx = np.zeros([img.shape[0], img.shape[1]])
    for label in labels:
        coords = np.where(np.all(img == np.array(label.color), axis=2))
        idx[coords[0], coords[1]] = label.id
    return idx


# kitti TF
def load_data(data_dir, mode='training'):
    images = list()
    labels = list()
    image_2_path = os.path.join(data_dir, mode, 'image_2')
    semantic_rgb_path = os.path.join(data_dir, mode, 'semantic_rgb')
    for root, dirs, files in os.walk(image_2_path):
        for file in files:
            if file.endswith(".png"):
                img = os.path.join(root, file)
                gt = os.path.join(semantic_rgb_path, file)
                images.append(img)
                labels.append(gt)
    return images, labels


def train_parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    gt_image_string = tf.io.read_file(label)
    image = tf.image.decode_png(image_string, channels=3)
    gt_image = tf.image.decode_png(gt_image_string, channels=3)
    # note that tf argument ordering is (h, w)
    resized_image = tf.image.resize(image, [256, 512])
    resized_gt_image = tf.image.resize(gt_image, [256, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return resized_image, resized_gt_image


def train_preprocess(image, gt_image):
    # image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # Make sure the image is still in [0, 1]
    # image = tf.clip_by_value(image, 0.0, 1.0)
    gt_image_idx = rgb_to_idx_tf(gt_image)
    return image, gt_image, gt_image_idx


def rgb_to_idx_tf(img):
    semantic_map = []
    for color in color_to_label:
        class_map = tf.reduce_all(tf.equal(img, color), axis=-1)
        semantic_map.append(class_map)
    semantic_map = tf.stack(semantic_map, axis=-1)
    # NOTE cast to tf.float32 because most neural networks operate in float32.
    semantic_map = tf.cast(semantic_map, tf.float32)
    class_indexes = tf.argmax(semantic_map, axis=-1)
    # NOTE this operation flattens class_indexes
    # class_indexes = tf.reshape(class_indexes, [-1])
    # color_image = tf.gather(palette, class_indexes)
    # color_image = tf.reshape(color_image, [img.shape[0], img.shape[1], 3])
    return class_indexes


def idx_to_rgb_tf(x):
    pass
