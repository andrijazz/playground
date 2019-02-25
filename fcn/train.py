#!/usr/bin/env python

"""
TODO:
* resize images during the training
* metrics
* Readme with results / playground readme about general guidelines
* plot predict.py (overlay)
* continue with training
"""
from __future__ import division, absolute_import, print_function

import argparse
import json
import tqdm

from settings import *
from model import *
from datasets.dataloader import *


def validate(sess, model, summary_op, summary_writer, val_set, step, config):
    """Validates on a single batch"""
    x_batch, y_batch, end_of_epoch = val_set.load_batch(batch_size=config.batch_size)
    feed = {
        model.x: x_batch,
        model.y: y_batch
    }

    summary_str = sess.run(summary_op, feed_dict=feed)
    summary_writer.add_summary(summary_str, global_step=step)


def train(config):
    # seeds
    # seed = 5
    # np.random.seed(seed)
    # tf.set_random_seed(seed)

    # dirs
    run_str = utils.make_hparam_string(vars(config))
    OUT_DIR = LOG_DIR + "/" + run_str
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # save config
    with open(OUT_DIR + "/" + CONFIG_FILENAME, 'w') as fp:
        json.dump(vars(config), fp)

    # logger
    logger = utils.get_logger(LOG_FILENAME, OUT_DIR)
    logger.info("Things are good ... training")
    logger.info("Configuration = {}".format(config))

    # init dataset
    train_set, val_set, _ = load(config.dataset, DATASET_DIR)

    # init model
    fcn_params = fcn_parameters(
        image_height=train_set.image_height,
        image_width=train_set.image_width,
        labels=train_set.labels,
        num_labels=train_set.num_labels,
        keep_prob=config.keep_prob,
        learning_rate=config.learning_rate,
        type=config.model)
    model = fcn(fcn_params)

    # init sess
    session_config = tf.ConfigProto()
    sess = tf.Session(config=session_config)

    # init summaries
    summary_writer = tf.summary.FileWriter(OUT_DIR)
    summary_writer.add_graph(sess.graph)
    summary_train_op = tf.summary.merge_all('train_collection')
    summary_val_op = tf.summary.merge_all('val_collection')

    # init saver
    saver = tf.train.Saver()

    # init vars
    init_vars = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    sess.run(init_vars)

    # init weights
    if config.init_weights:
        sess.run(model.initialize_weights_op())

    logger.info("Training started")

    epoch = 1
    step = 1
    pbar = tqdm.tqdm(total=config.num_epochs * len(train_set.file_pairs) / config.batch_size)
    while epoch <= config.num_epochs:
        x_batch, y_batch, end_of_epoch = train_set.load_batch(batch_size=config.batch_size)
        feed = {
            model.x: x_batch,
            model.y: y_batch
        }
        sess.run(model.goal, feed_dict=feed)
        if end_of_epoch:
            epoch += 1

        if step % 10 == 0:
            summary_str = sess.run(summary_train_op, feed_dict=feed)
            summary_writer.add_summary(summary_str, global_step=step)
        if step % 50 == 0:
            validate(sess, model, summary_val_op, summary_writer, val_set, step, config)
        if step % 10000 == 0:
            saver.save(sess, OUT_DIR + "/" + MODEL_NAME, global_step=step)

        if debug:
            summary_str = sess.run(summary_train_op, feed_dict=feed)
            summary_writer.add_summary(summary_str, global_step=step)
            validate(sess, model, summary_val_op, summary_writer, val_set, step, config)
            break

        pbar.update(1)
        step += 1

    pbar.close()
    saver.save(sess, OUT_DIR + "/" + MODEL_NAME, global_step=step)

    # clean up tf things
    sess.close()
    tf.reset_default_graph()

    return run_str


def main(_):
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fully Convolutional Networks TensorFlow implementation [Training]')

    parser.add_argument('-m', '--model', type=str, help='model. fcn32 or fcn16 or fcn8', default='fcn32')
    parser.add_argument('-d', '--dataset', type=str, help='dataset to train on. kitti or cityscapes', default='kitti')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=2)
    parser.add_argument('-n', '--num_epochs', type=int, help='number of epochs', default=50)
    parser.add_argument('-l', '--learning_rate', type=float, help='learning rate', default=1e-3)
    parser.add_argument('-g', '--gpu', type=int, help='GPU to use for training', default=1)
    parser.add_argument('-k', '--keep_prob', type=float, help='dropout keep prob. default is 0.5.', default=float(0.5))
    parser.add_argument('-i', '--init_weights', type=bool, help='use pretrained vgg-16 weights', default=True)
    args = parser.parse_args()

    tf.app.run()
