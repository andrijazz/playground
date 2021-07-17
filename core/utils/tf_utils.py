import tensorflow as tf


def rgb_to_idx_tf(rgb_img, color_to_label_map):
    semantic_map = []
    for color in color_to_label_map:
        class_map = tf.reduce_all(tf.equal(rgb_img, color), axis=-1)
        semantic_map.append(class_map)
    semantic_map = tf.stack(semantic_map, axis=-1)
    # NOTE cast to tf.float32 because most neural networks operate in float32.
    semantic_map = tf.cast(semantic_map, tf.float32)
    class_indexes = tf.argmax(semantic_map, axis=-1)
    return class_indexes


def idx_to_rgb_tf(idx_img, idx_to_color_map):
    palette = tf.constant(idx_to_color_map, dtype=tf.uint8)
    # NOTE this operation flattens class_indexes
    class_indexes = tf.reshape(idx_img, [-1])
    color_image = tf.gather(palette, class_indexes)
    color_image = tf.reshape(color_image, [idx_img.shape[0], idx_img.shape[1], 3])
    return color_image


def idx_to_rgb_batch_tf(idx_imgs, idx_to_color_map):
    color_images = []
    for i in range(idx_imgs.shape[0]):
        color_images.append(idx_to_rgb_tf(idx_imgs[i], idx_to_color_map))
    color_images = tf.stack(color_images, axis=0)
    return color_images
