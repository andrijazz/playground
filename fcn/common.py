import numpy as np

c_image_height = 370
c_image_width = 1224
classes = np.array(
    [[  0,   0,   0],
       [  0,   0,  70],
       [  0,   0,  90],
       [  0,   0, 110],
       [  0,   0, 142],
       [  0,   0, 230],
       [  0,  60, 100],
       [  0,  80, 100],
       [ 70,  70,  70],
       [ 70, 130, 180],
       [ 81,   0,  81],
       [102, 102, 156],
       [107, 142,  35],
       [111,  74,   0],
       [119,  11,  32],
       [128,  64, 128],
       [150, 100, 100],
       [150, 120,  90],
       [152, 251, 152],
       [153, 153, 153],
       [180, 165, 180],
       [190, 153, 153],
       [220,  20,  60],
       [220, 220,   0],
       [230, 150, 140],
       [244,  35, 232],
       [250, 170,  30],
       [250, 170, 160],
       [255,   0,   0]])


probability_classes = {
       tuple([  0,   0,   0]): np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([  0,   0,  70]): np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([  0,   0,  90]): np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([  0,   0, 110]): np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([  0,   0, 142]): np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([  0,   0, 230]): np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([  0,  60, 100]): np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([  0,  80, 100]): np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([ 70,  70,  70]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([ 70, 130, 180]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([ 81,   0,  81]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([102, 102, 156]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([107, 142,  35]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([111,  74,   0]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([119,  11,  32]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([128,  64, 128]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([150, 100, 100]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([150, 120,  90]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([152, 251, 152]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([153, 153, 153]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([180, 165, 180]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
       tuple([190, 153, 153]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
       tuple([220,  20,  60]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
       tuple([220, 220,   0]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
       tuple([230, 150, 140]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
       tuple([244,  35, 232]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
       tuple([250, 170,  30]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
       tuple([250, 170, 160]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
       tuple([255,   0,   0]): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
}

idx_classes = {
    tuple([  0,   0,   0]): 0,
    tuple([  0,   0,  70]): 1,
    tuple([  0,   0,  90]): 2,
    tuple([  0,   0, 110]): 3,
    tuple([  0,   0, 142]): 4,
    tuple([  0,   0, 230]): 5,
    tuple([  0,  60, 100]): 6,
    tuple([  0,  80, 100]): 7,
    tuple([ 70,  70,  70]): 8,
    tuple([ 70, 130, 180]): 9,
    tuple([ 81,   0,  81]): 10,
    tuple([102, 102, 156]): 11,
    tuple([107, 142,  35]): 12,
    tuple([111,  74,   0]): 13,
    tuple([119,  11,  32]): 14,
    tuple([128,  64, 128]): 15,
    tuple([150, 100, 100]): 16,
    tuple([150, 120,  90]): 17,
    tuple([152, 251, 152]): 18,
    tuple([153, 153, 153]): 19,
    tuple([180, 165, 180]): 20,
    tuple([190, 153, 153]): 21,
    tuple([220,  20,  60]): 22,
    tuple([220, 220,   0]): 23,
    tuple([230, 150, 140]): 24,
    tuple([244,  35, 232]): 25,
    tuple([250, 170,  30]): 26,
    tuple([250, 170, 160]): 27,
    tuple([255,   0,   0]): 28
}

num_classes = 29
DATA_DIR = "../data"
LOG_DIR = "../log"
