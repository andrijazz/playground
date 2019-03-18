import argparse
import json
import tqdm
import vapk as utils
import numpy as np

from monodepth.settings import *
from monodepth.model import *
from datasets.dataloader import *
from utils import metrics


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
    _, _, test_set = load("kitti_scene_flow", DATASET_DIR)

    # init sess
    session_config = tf.ConfigProto()
    sess = tf.Session(config=session_config)

    # init model
    monodepth_params = monodepth_parameters(
        image_height=test_set.image_height,
        image_width=test_set.image_width,
        in_channels=test_set.num_channels,
        learning_rate=config.learning_rate,
        alpha_image_loss=config.alpha_image_loss,
        disp_gradient_loss_weight=config.disp_loss,
        lr_loss_weight=config.lr_loss
    )
    model = Monodepth(monodepth_params)

    # init summaries
    summary_writer = tf.summary.FileWriter(OUT_DIR)
    summary_writer.add_graph(sess.graph)
    summary_test_op = tf.summary.merge_all('val_collection')

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
    pbar = tqdm.tqdm(total=len(test_set.file_pairs))

    rse = list()
    abs_rel = list()
    rmse = list()
    rmse_log = list()
    while True:
        left_batch, right_batch, gt_batch, end_of_epoch = test_set.load_batch(batch_size=1)
        feed = {
            model.left: left_batch
        }

        summary_str = sess.run(summary_test_op, feed_dict=feed)
        summary_writer.add_summary(summary_str, global_step=step)

        rse.extend(metrics.rse(left_batch, gt_batch))
        abs_rel.extend(metrics.abs_rel(left_batch, gt_batch))
        rmse.extend(metrics.rmse(left_batch, gt_batch))
        rmse_log.extend(metrics.rmse_log(left_batch, gt_batch))

        if end_of_epoch:
            break

        if step == debug_steps:
            break

        step += 1
        pbar.update(1)
    pbar.close()

    result = np.zeros(len(test_set.file_pairs), 4)
    result[:, 0] = rse
    result[:, 1] = abs_rel
    result[:, 2] = rmse
    result[:, 3] = rmse_log

    np.save(OUT_DIR + "/result.npy", result)

    # clean up tf things
    sess.close()
    tf.reset_default_graph()


def main(_):
    test(args.run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monodepth - TensorFlow implementation [Testing]')
    parser.add_argument('-r', '--run', type=str, help='run', required=True)
    args = parser.parse_args()

    tf.app.run()
