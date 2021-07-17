import os
import shutil
import sys
import numpy as np

import requests
import torchvision
import torchvision.transforms as transforms
import wandb
import yaml
from PIL import Image


def is_debug_session():
    gettrace = getattr(sys, 'gettrace', None)
    debug_session = not ((gettrace is None) or (not gettrace()))
    return debug_session


# TODO remove tensorboard plot
def plot(items, tag, data_path, step, log_to_wandb=True, writer=None):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    imgs = list()
    for img_file in items:
        path = os.path.join(data_path, 'images', '%s.jpg' % img_file)
        img = transform(Image.open(path).convert('RGB'))
        imgs.append(img)

    if log_to_wandb:
        wandb_imgs = list()
        for i in imgs:
            wandb_imgs.append(wandb.Image(i))
        wandb.log({tag: wandb_imgs})
    else:
        grid = torchvision.utils.make_grid(imgs, nrow=5)
        writer.add_image(tag, grid, global_step=step)


def download_image(url, output_file):
    response = None
    try:
        response = requests.get(url, stream=True)
    except Exception as e:
        print(e)

    if response is None:
        return None

    if response.status_code != 200:
        return None

    with open(output_file, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    return output_file


def pad_to_square(img):
    old_size = img.size
    m = max(old_size[0], old_size[1])
    new_size = (m, m)
    padding_color = (255, 255, 255)
    new_im = Image.new("RGB", new_size, padding_color)
    new_im.paste(img, ((new_size[0] - old_size[0]) // 2, (new_size[1] - old_size[1]) // 2))
    return new_im


def get_config_yml(yml_config_file):
    try:
        with open(yml_config_file) as file:
            config_dict = yaml.load(file, Loader=yaml.FullLoader)
            return config_dict['config']
    except FileNotFoundError:
        print('Config file {} not found'.format(yml_config_file))
        exit(1)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def rgb_to_one_hot(img, labels):
    probability = np.zeros([img.shape[0], img.shape[1], len(labels)])
    for label in labels:
        coords = np.where(np.all(img == np.array(label.color), axis=2))
        one_hot = np.zeros(len(labels))
        one_hot[label.id] = 1
        probability[coords[0], coords[1], :] = one_hot
    return probability


def rgb_to_idx(img, labels):
    idx = np.zeros([img.shape[0], img.shape[1]])
    for label in labels:
        coords = np.where(np.all(img == np.array(label.color), axis=2))
        idx[coords[0], coords[1]] = label.id
    return idx


def find_module_path(module_name):
    src_root = os.getenv('SRC_ROOT')
    module_paths = list(Path(src_root).rglob(module_name + ".py"))
    assert len(module_paths) == 1, "There are several modules named \"{}\" which is not by our convention. Please" \
                                   "rename your module."
    relpath = os.path.relpath(str(module_paths[0]), src_root)
    relpath = relpath.split('.')[0]     # strip .py ext
    relpath = relpath.replace("/", ".")
    return relpath


def camel_case_to_snake_case(s):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
