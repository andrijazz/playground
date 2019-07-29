#!/usr/bin/env python

"""
TODO:
* new utils
* metrics
* Readme with results / playground readme about general guidelines
* plot predict.py (overlay)
* multiple gpu's
"""

from __future__ import division, absolute_import, print_function

import argparse
import json
import os

from fcn.model import *
from datasets.dataloader import *
from utils.utils_n import *


def init_dirs():
    """
    Makes sure that all dirs, files, args are in place.
    :return:
    """
    # this is abs path of directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # root dir is always parent dir to model files
    root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

    # check if project root folder is in $PYTHONPATH
    python_path_string = os.getenv('PYTHONPATH')
    if root_dir not in python_path_string.split(':'):
        exit("ROOT dir of the project is not in PYTHONPATH."
             "Try adding following command in /etc/.profile file and restarting the computer after the changes: "
             "PYTHONPATH=${PYTHONPATH}:<path_to_your_project>")

    # model base name
    args.name = "fcn"

    # dataset dir
    args.dataset_dir = os.path.join(args.data_drive, "datasets")

    # create run_str
    args.run = make_hparam_string(vars(args), ignored_keys=["name", "dataset_dir", "data_drive", "resume", "weights", "val_step", "save_step", "debug_steps"])

    # log dir
    args.log_dir = os.path.join(args.data_drive, "log", args.name, args.run)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # log file - commenting out since its handle inside logger
    # config.log_file = os.path.join(config.log_dir, "{}.log".format(config.name))

    # config files are always config.json
    args.config_filename = os.path.join(args.log_dir, "config.json")

    # checkpoint files are always named based on args.name
    args.checkpoint_filename = os.path.join(args.log_dir, args.name)

    # save configuration
    with open(args.config_filename, 'w') as fp:
        json.dump(vars(args), fp)


def validate(sess, val_model, summary_writer, step):
    sess.run(val_model.iterator_init_op)
    try:
        while True:
            summary_str = sess.run(val_model.summary_op)
            summary_writer.add_summary(summary_str, global_step=step)
            # just write one image from val
            # delete this break in case you want validation on whole val set
            break
    except tf.errors.OutOfRangeError:
        return


def train(config):
    # logger
    # TODO: logger should take only log file as an arg and all (refactor) when removing vapk
    logger = utils.get_logger(config.name, config.log_dir)
    logger.info("Things are good ... training")
    logger.info("Configuration = {}".format(config))

    # init datasets
    train_set, val_set, _ = load(config.dataset, config.dataset_dir)

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
            type=TRAIN
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
            type=VAL
        ),
        reuse_variables=tf.AUTO_REUSE
    )

    # init sess
    session_config = tf.ConfigProto()
    sess = tf.Session(config=session_config)

    # init summaries
    summary_writer = tf.summary.FileWriter(config.log_dir)
    summary_writer.add_graph(sess.graph)

    # init saver
    saver = tf.train.Saver()

    # init vars
    init_vars = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    sess.run(init_vars)

    # restore model
    if config.resume:
        saver.restore(sess, config.resume)
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

                # debug steps
                if step <= config.debug_steps:
                    summary_str = sess.run(train_model.summary_op)
                    summary_writer.add_summary(summary_str, global_step=step)
                    validate(sess, val_model, summary_writer, step)
                    saver.save(sess, config.checkpoint_filename, global_step=step)
                else:
                    if step % config.val_step == 0:
                        summary_str = sess.run(train_model.summary_op)
                        summary_writer.add_summary(summary_str, global_step=step)
                        validate(sess, val_model, summary_writer, step)

                    if step % config.save_step == 0:
                        saver.save(sess, config.checkpoint_filename, global_step=step)

                step += 1

        except tf.errors.OutOfRangeError:
            logger.info("End of epoch {} (step = {})".format(epoch, step))

    saver.save(sess, config.checkpoint_filename, global_step=step)

    # clean up tf things
    sess.close()
    tf.reset_default_graph()


def main(_):
    init_dirs()
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fully Convolutional Networks TensorFlow implementation [Training]')
    parser.add_argument('--data_drive', type=str, help='Data drive', required=True)
    parser.add_argument('--model', type=str, choices=["fcn32", "fcn16", "fcn8"], help='Model version', required=True)
    parser.add_argument('--dataset', type=str, choices=["kitti_semantics", "cityscapes"], help='Dataset', required=True)
    parser.add_argument('--batch_size', type=int, help='Batch size', required=True)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs', required=True)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=1e-3)
    parser.add_argument('--keep_prob', type=float, help='Dropout keep prob', default=0.5)
    parser.add_argument('--resume', type=str, help='Checkpoint full path. Empty string if training from the beginning', default='')
    parser.add_argument('--weights', type=str, help='Path to file with pre-trained vgg-16 weights', default='vgg16_weights.npz')
    parser.add_argument('--gpu', type=int, help='GPU to use for training', default=0)
    parser.add_argument('--debug_steps', type=int, help='Number of debug steps', default=10)
    parser.add_argument('--save_step', type=int, help='Save model on every x-th step', default=1000)
    parser.add_argument('--val_step', type=int, help='Validate model and log summaries on every x-th step', default=1000)

    args = parser.parse_args()
    tf.app.run()
