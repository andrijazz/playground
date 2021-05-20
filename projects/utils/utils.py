import numpy as np
import yaml


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


def load_config(yml_config_file):
    try:
        with open(yml_config_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            return config['config']
    except FileNotFoundError:
        print('Config file {} not found'.format(yml_config_file))
        exit(1)
