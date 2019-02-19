import argparse
import json
from model import *
from dataloader import *


def test(run):
    # seeds
    # seed = 5
    # np.random.seed(seed)
    # tf.set_random_seed(seed)

    # dirs
    OUT_DIR = LOG_DIR + "/" + run
    if not os.path.exists(OUT_DIR):
        exit("Run {} doesn't exist".format(run))

    # logger
    logger = utils.get_logger(LOG_FILENAME, OUT_DIR)
    logger.info("Things are good ... testing")
    logger.info("Configuration = {}".format(run))

    # load config
    with open(OUT_DIR + "/" + CONFIG_FILENAME, 'r') as fp:
        config = json.load(fp)

    # convert dict to argparse namespace
    config = namedtuple('parameters', config.keys())(**config)

    # init dataset
    _, _, test_set = load(config.dataset)

    # init sess
    session_config = tf.ConfigProto()
    sess = tf.Session(config=session_config)

    # init model
    params = fcn_parameters(
        image_height=test_set.image_height,
        image_width=test_set.image_width,
        labels=test_set.labels,
        num_labels=test_set.num_labels,
        keep_prob=config.keep_prob,
        learning_rate=config.learning_rate,
        type=config.model)
    model = fcn(params)

    # init summaries
    summary_writer = tf.summary.FileWriter(OUT_DIR)
    summary_writer.add_graph(sess.graph)
    summary_test_op = tf.summary.merge_all('test_collection')

    # init saver
    saver = tf.train.Saver()

    # init vars
    init_vars = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    sess.run(init_vars)

    # restore model
    restore_path = tf.train.latest_checkpoint(OUT_DIR)
    saver.restore(sess, restore_path)

    logger.info("Testing started")
    step = 1
    while True:
        x_batch, _, end_of_epoch = test_set.load_batch(batch_size=1)    # test batch size is always 1
        feed = {
            model.x: x_batch,
        }
        sess.run(model.output, feed_dict=feed)
        if end_of_epoch:
            break

        summary_str = sess.run(summary_test_op, feed_dict=feed)
        summary_writer.add_summary(summary_str, global_step=step)
        step += 1

        if debug:
            break

    # TODO save iou / ppa
    # np.save(OUT_DIR + "/results.npy", results)

    # clean up tf things
    sess.close()
    tf.reset_default_graph()


def main(_):
    test(args.run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fully Convolutional Networks TensorFlow implementation [Testing]')
    parser.add_argument('-r', '--run', type=str, help='run', required=True)
    args = parser.parse_args()

    tf.app.run()
