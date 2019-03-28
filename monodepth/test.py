import argparse
import json
import tqdm
import vapk as utils
import numpy as np
import cv2
import scipy.misc

from monodepth.settings import *
from monodepth.model import *
from datasets.dataloader import *

# kitti
width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def convert_disps_to_depths_kitti(gt_disparities, pred_disparities):
    gt_depths = []
    pred_depths = []
    pred_disparities_resized = []

    for i in range(len(gt_disparities)):
        gt_disp = gt_disparities[i]
        height, width = gt_disp.shape

        pred_disp = pred_disparities[i]
        pred_disp = width * cv2.resize(pred_disp, (width, height), interpolation=cv2.INTER_LINEAR)

        pred_disparities_resized.append(pred_disp)

        mask = gt_disp > 0

        gt_depth = width_to_focal[width] * 0.54 / (gt_disp + (1.0 - mask))
        pred_depth = width_to_focal[width] * 0.54 / pred_disp

        gt_depths.append(gt_depth)
        pred_depths.append(pred_depth)
    return gt_depths, pred_depths, pred_disparities_resized


def test(args):
    # seeds
    # seed = 5
    # np.random.seed(seed)
    # tf.set_random_seed(seed)

    # dirs
    OUT_DIR = LOG_DIR + "/" + args.run
    if not os.path.exists(OUT_DIR):
        exit("Run {} doesn't exist".format(args.run))

    # logger
    logger = utils.get_logger(LOG_FILENAME, OUT_DIR)
    logger.info("Things are good ... testing")
    logger.info("Configuration = {}".format(args.run))

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
        alpha_image_loss=config.alpha_image_loss,
        disp_gradient_loss_weight=config.disp_loss,
        lr_loss_weight=config.lr_loss
    )
    model = Monodepth(monodepth_params)

    # init saver
    saver = tf.train.Saver()

    # init vars
    init_vars = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    sess.run(init_vars)

    # restore model
    if not args.checkpoint:
        restore_path = tf.train.latest_checkpoint(OUT_DIR)
    else:
        restore_path = OUT_DIR + "/" + args.checkpoint

    saver.restore(sess, restore_path)

    logger.info("Testing started")
    num_samples = len(test_set.file_pairs)
    pbar = tqdm.tqdm(total=num_samples)

    # rse      = np.zeros(num_samples, np.float32)
    # abs_rel  = np.zeros(num_samples, np.float32)
    # rmse     = np.zeros(num_samples, np.float32)
    # rmse_log = np.zeros(num_samples, np.float32)
    results_dir = OUT_DIR + "/results"
    c = 0

    gt_disparities = []
    pred_disparities = []
    while True:
        left, right, gt, end_of_epoch = test_set.load_batch(batch_size=1)
        feed = {
            model.left: left
        }

        est = sess.run(model.h_disp1, feed_dict=feed)
        left_est = est[:, :, :, 0]
        pred_disparities.append(left_est[0])
        gt_disparities.append(gt[0])

        # save images
        scipy.misc.imsave(results_dir + "/" + str(c) + "_left.png", left[0])
        # scipy.misc.imsave(results_dir + "/" + str(i) + "_gt.png", gt[0])
        scipy.misc.imsave(results_dir + "/" + str(c) + "_disp.png", left_est[0])

        if end_of_epoch:
            break
        c += 1
        pbar.update(1)
    pbar.close()

    # result = np.zeros((4, num_samples), np.float32)
    # result[0] = rse
    # result[1] = abs_rel
    # result[2] = rmse
    # result[3] = rmse_log
    # np.save(results_dir + "/result.npy", result)

    gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_kitti(gt_disparities, pred_disparities)

    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1_all = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)

    min_depth = 1e-3
    max_depth = 80
    for i in range(num_samples):
        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth

        gt_disp = gt_disparities[i]
        mask = gt_disp > 0
        pred_disp = pred_disparities_resized[i]

        disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
        bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
        d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))

    # clean up tf things
    sess.close()
    tf.reset_default_graph()


def main(_):
    test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monodepth - TensorFlow implementation [Testing]')
    parser.add_argument('-r', '--run', type=str, help='run', required=True)
    parser.add_argument('-c', '--checkpoint', type=str, help='checkpoint', default="")
    args = parser.parse_args()

    tf.app.run()
