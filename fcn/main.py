#!/usr/bin/env python
import argparse
import os
import datasets.loader
from model import *


# ---------------------------------------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Fully Convolutional Networks TensorFlow implementation.')

parser.add_argument('--model_name',         type=str,               help='model name',                                      default='fcn32')
parser.add_argument('--dataset',            type=str,               help='dataset to train on, kitti, or cityscapes',       default='kitti')
parser.add_argument('--batch_size',         type=int,               help='batch size',                                      default=2)
parser.add_argument('--num_epochs',         type=int,               help='number of epochs',                                default=50)
parser.add_argument('--learning_rate',      type=float,             help='initial learning rate',                           default=1e-3)
parser.add_argument('--gpu',                type=int,               help='GPU to use for training',                         default=1)
parser.add_argument('--num_threads',        type=int,               help='number of threads to use for data loading',       default=8)
parser.add_argument('--keep_prob',          type=float,             help='dropout keep prob. Default is 0.5.',              default=float(0.5))
parser.add_argument('--vgg-16',             type=bool,              help='use pretrained vgg-16 weights',                   default=False)
args = parser.parse_args()

# seeds
# seed = 5
# np.random.seed(seed)
# tf.set_random_seed(seed)


# dirs
LOG_DIR = "../log"
run_str = utils.make_hparam_string(vars(args))
OUT_DIR = LOG_DIR + "/" + run_str
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# logger
logger = utils.setup_logger("fcn", OUT_DIR)
logger.info("Things are good!")
logger.info("Configuration = {}".format(args))

# init dataset
train_set, val_set, test_set = datasets.loader.get(args.dataset)

# init model
params = fcn_parameters(
    image_height=train_set.image_height,
    image_width=train_set.image_width,
    labels=train_set.labels,
    num_labels=train_set.num_labels,
    keep_prob=args.keep_prob,
    learning_rate=args.learning_rate)
model = fcn32(params)

# init sess
session_config = tf.ConfigProto()
sess = tf.Session(config=session_config)
# init summaries
summary_writer = tf.summary.FileWriter(OUT_DIR)
summary_writer.add_graph(sess.graph)
summary_train_op = tf.summary.merge_all('train_collection')
summary_val_op = tf.summary.merge_all('val_collection')
summary_test_op = tf.summary.merge_all('test_collection')
# init vars
init_vars = [tf.global_variables_initializer(), tf.local_variables_initializer()]
sess.run(init_vars)


def validate(step):
    """Validates on a single batch"""
    x_batch, y_batch, end_of_epoch = val_set.load_batch(batch_size=args.batch_size)
    feed = {
        model.x: x_batch,
        model.y: y_batch
    }

    summary_str = sess.run(summary_val_op, feed_dict=feed)
    summary_writer.add_summary(summary_str, global_step=step)


def train():
    logger.info("training started")

    epoch = 1
    step = 1
    while epoch <= args.num_epochs:
        x_batch, y_batch, end_of_epoch = train_set.load_batch(batch_size=args.batch_size)
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
            validate(step)

        step += 1
        break


def test():
    logger.info("testing started")
    step = 1
    while True:
        x_batch, _, end_of_epoch = test_set.load_batch(batch_size=args.batch_size)
        feed = {
            model.x: x_batch,
        }
        sess.run(model.output, feed_dict=feed)
        if end_of_epoch:
            break

        summary_str = sess.run(summary_test_op, feed_dict=feed)
        summary_writer.add_summary(summary_str, global_step=step)
        step += 1

    # TODO save iou / ppa
    # np.save(OUT_DIR + "/results.npy", results)


def main(_):
    train()
    test()


if __name__ == "__main__":
    tf.app.run()
