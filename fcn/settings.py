import os

FCN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(FCN_DIR, os.pardir))
DATASET_DIR = ROOT_DIR + "/datasets"
LOG_DIR = ROOT_DIR + "/log"
MODEL_NAME = "fcn"
LOG_FILENAME = "fcn"
CONFIG_FILENAME = "config.json"
debug = True
