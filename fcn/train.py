#!/usr/bin/env python

"""
TODO:
* continue with training
* new utils
* metrics
* Readme with results / playground readme about general guidelines
* plot predict.py (overlay)
* multiple gpu's
* rename test to eval
"""

from __future__ import division, absolute_import, print_function

import argparse
import json

from fcn.settings import *
from fcn.model import *
from datasets.dataloader import *
from utils.utils_n import *


def validate(sess, val_model, summary_writer, step):
    sess.run(val_model.iterator_init_op)
    try:
        while True:
            summary_str = sess.run(val_model.summary_op)
            summary_writer.add_summary(summary_str, global_step=step)
            break # just write one image from val
    except tf.errors.OutOfRangeError:
        return


def train(config):
    # seeds
    # seed = 5
    # np.random.seed(seed)
    # tf.set_random_seed(seed)

    # dirs
    run_str = make_hparam_string(vars(config), ignored_keys=["checkpoint", "weights", "log_on", "save_on", "debug"])
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
            model=config.model,
            type=TRAINING
        ),
        reuse_variables=tf.AUTO_REUSE
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
            model=config.model,
            type=EVAL
        ),
        reuse_variables=tf.AUTO_REUSE
    )

    # init sess
    session_config = tf.ConfigProto()
    sess = tf.Session(config=session_config)

    # init summaries
    summary_writer = tf.summary.FileWriter(OUT_DIR)
    summary_writer.add_graph(sess.graph)

    # init saver
    saver = tf.train.Saver()

    # init vars
    init_vars = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    sess.run(init_vars)

    # restore model
    if config.checkpoint:
        saver.restore(sess, config.checkpoint)
    else:
        # init weights only if we are not restoring the model and config.weights is set
        if config.weights:
            sess.run(train_model.initialize_weights_op(config.weights))

    logger.info("Training started")

    step = 1
    for epoch in range(config.num_epochs):
        sess.run(train_model.iterator_init_op)
        try:
            while True:
                sess.run(train_model.train_op)

                if step % config.log_on == 0:
                    summary_str = sess.run(train_model.summary_op)
                    summary_writer.add_summary(summary_str, global_step=step)
                    validate(sess, val_model, summary_writer, step)

                if step % config.save_on == 0:
                    saver.save(sess, OUT_DIR + "/" + MODEL_NAME, global_step=step)

                # debug steps
                if step <= config.debug:
                    summary_str = sess.run(train_model.summary_op)
                    summary_writer.add_summary(summary_str, global_step=step)
                    validate(sess, val_model, summary_writer, step)
                    saver.save(sess, OUT_DIR + "/" + MODEL_NAME, global_step=step)

                step += 1

        except tf.errors.OutOfRangeError:
            logger.info("End of epoch. step = {}".format(step))

    saver.save(sess, OUT_DIR + "/" + MODEL_NAME, global_step=step)

    # clean up tf things
    sess.close()
    tf.reset_default_graph()

    return run_str


def main(_):
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fully Convolutional Networks TensorFlow implementation [Training]')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint full path', required=True)
    parser.add_argument('--model', type=str, help='Model type (fcn32, fcn16 or fcn8)', required=True)
    parser.add_argument('--dataset', type=str, help='Dataset (kitti_semantics or cityscapes)', required=True)
    parser.add_argument('--batch_size', type=int, help='Batch size', required=True)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs', required=True)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', required=True)
    parser.add_argument('--gpu', type=int, help='GPU to use for training', required=True)
    parser.add_argument('--keep_prob', type=float, help='Dropout keep prob', required=True)
    parser.add_argument('--weights', type=str, help='Path to file with pre-trained vgg-16 weights', required=True)
    parser.add_argument('--save_on', type=int, help='Save model on x-th step', required=True)
    parser.add_argument('--log_on', type=int, help='Log summaries on x-th step', required=True)
    parser.add_argument('--debug', type=int, help='Number of debug steps', required=True)
    args = parser.parse_args()

    tf.app.run()
