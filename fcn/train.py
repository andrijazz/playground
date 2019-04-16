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

from fcn.settings import *
from fcn.model import *
from datasets.dataloader import *


def validate(sess, val_model, summary_val_op, summary_writer, step):
    sess.run(val_model.iterator_init_op)
    try:
        while True:
            summary_str = sess.run(summary_val_op)
            summary_writer.add_summary(summary_str, global_step=step)
    except tf.errors.OutOfRangeError:
        return


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

    # init datasets
    train_set, val_set, _ = load(config.dataset, DATASET_DIR)

    # init models
    train_model = fcn(
        fcn_parameters(
            batch_size=config.batch_size,
            data=train_set,
            image_height=256,
            image_width=512,
            labels=train_set.labels,
            num_labels=train_set.num_labels,
            keep_prob=config.keep_prob,
            learning_rate=config.learning_rate,
            type=config.model
        )
    )

    val_model = fcn(
        fcn_parameters(
            batch_size=config.batch_size,
            data=val_set,
            image_height=256,
            image_width=512,
            labels=val_set.labels,
            num_labels=val_set.num_labels,
            keep_prob=config.keep_prob,
            learning_rate=config.learning_rate,
            type=config.model
        )
    )

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
    # if config.init_weights:
    #     sess.run(model.initialize_weights_op())

    logger.info("Training started")

    # total_steps = config.num_epochs * train_set.num_instances // config.batch_size
    pbar = tqdm.tqdm(total=config.num_epochs)
    step = 1
    for epoch in range(config.num_epochs):

        sess.run(train_model.iterator_init_op)
        try:
            while True:
                sess.run(train_model.train_op)

                if step % 100 == 0:
                    summary_str = sess.run(summary_train_op)
                    summary_writer.add_summary(summary_str, global_step=step)
                if step % 1000 == 0:
                    validate(sess, val_model, summary_val_op, summary_writer, step)
                if step % 10000 == 0:
                    saver.save(sess, OUT_DIR + "/" + MODEL_NAME, global_step=step)
                if debug:
                    validate(sess, val_model, summary_val_op, summary_writer, step)
                    # break

                step += 1

        except tf.errors.OutOfRangeError:
            pbar.update(1)

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
    parser.add_argument('-d', '--dataset', type=str, help='dataset to train on. kitti or cityscapes', default='kitti_semantics')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=2)
    parser.add_argument('-n', '--num_epochs', type=int, help='number of epochs', default=50)
    parser.add_argument('-l', '--learning_rate', type=float, help='learning rate', default=1e-3)
    parser.add_argument('-g', '--gpu', type=int, help='GPU to use for training', default=1)
    parser.add_argument('-k', '--keep_prob', type=float, help='dropout keep prob. default is 0.5.', default=float(0.5))
    parser.add_argument('-i', '--init_weights', type=bool, help='use pretrained vgg-16 weights', default=True)
    args = parser.parse_args()

    tf.app.run()
