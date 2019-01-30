import argparse
import os
import scipy.misc
import itertools
import tqdm
import numpy as np
import vapk as utils

# ---------------------------------------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--instances_dir', type=str, help="""Path to instances data dir""", required=True)
parser.add_argument('--labels_dir', type=str, help="""Path to labels data dir""", required=True)
parser.add_argument('--dst_dir', type=str, help="""Path where prepared data will be stored""", required=True)

# transformations
# parser.add_argument('--resize', help="""Resize image to size""")
parser.add_argument('--resize_to_min_size',
                    type=bool,
                    help="""Resize all the images to the size of the smallest one""",
                    default=False)
# parser.add_argument('--center_crop', type=bool, help="""Center crop images""", default=False)
# parser.add_argument('--scaling', type=bool, help="""Scaling images""", default=False)
# parser.add_argument('--translation', type=bool, help="""Translation images""", default=False)
# parser.add_argument('--rotation', type=bool, help="""Translation images""", default=False)
parser.add_argument('--salt_and_paper', type=bool, help="""Apply salt and paper noise""", default=False)
# parser.add_argument('--shade', type=bool, help="""Apply shade""", default=False)
parser.add_argument('--mirror', type=bool, help="""Mirror images""", default=False)

config = parser.parse_args()

# ---------------------------------------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------------------------------------
if not os.path.exists(config.instances_dir) or not os.path.exists(config.labels_dir):
    exit("Configured data path doesn't exist")

if not os.path.exists(config.dst_dir):
    os.makedirs(config.dst_dir)

INSTANCES_DIR = config.dst_dir + "/x"
if not os.path.exists(INSTANCES_DIR):
    os.makedirs(INSTANCES_DIR)
LABELS_DIR = config.dst_dir + "/y"
if not os.path.exists(LABELS_DIR):
    os.makedirs(LABELS_DIR)

# ---------------------------------------------------------------------------------------------------------------------
# Apply transformations
# ---------------------------------------------------------------------------------------------------------------------
images = utils.semantic_segmentation.load_data(config.instances_dir, config.labels_dir)
all_image_files = list(itertools.chain.from_iterable(images))
min_h, min_w = utils.semantic_segmentation.calculate_min_image_size(all_image_files)
pbar = tqdm.tqdm(total=len(images))
for [image_file, gt_image_file] in images:
    x = scipy.misc.imread(image_file)
    x_basename = os.path.splitext(os.path.basename(image_file))[0]
    x_extension = os.path.splitext(os.path.basename(image_file))[1]

    y = scipy.misc.imread(gt_image_file)
    y_basename = os.path.splitext(os.path.basename(gt_image_file))[0]
    y_extension = os.path.splitext(os.path.basename(gt_image_file))[1]

    if config.resize_to_min_size:
        x = scipy.misc.imresize(x, (min_h, min_w), interp="bilinear")
        y = scipy.misc.imresize(y, (min_h, min_w), interp="nearest")
        scipy.misc.imsave(INSTANCES_DIR + "/" + x_basename + x_extension, x)
        scipy.misc.imsave(LABELS_DIR + "/" + y_basename + y_extension, y)

    if config.mirror:
        suffix = "_m"
        x_m = np.fliplr(x)
        y_m = np.fliplr(y)
        scipy.misc.imsave(INSTANCES_DIR + "/" + x_basename + suffix + x_extension, x_m)
        scipy.misc.imsave(LABELS_DIR + "/" + y_basename + suffix + y_extension, y_m)

    if config.salt_and_paper:
        suffix = "_sp"
        x_sp = utils.semantic_segmentation.salt_and_pepper_noise(x)
        scipy.misc.imsave(INSTANCES_DIR + "/" + x_basename + suffix + x_extension, x_sp)
        scipy.misc.imsave(LABELS_DIR + "/" + y_basename + suffix + y_extension, y)

    pbar.update(1)
pbar.close()

