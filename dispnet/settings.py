import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
sys.path.insert(0, ROOT_DIR)

DATASET_DIR = "/mnt/datasets"
MODEL_NAME = "dispnet"
LOG_DIR = "/mnt/log/" + MODEL_NAME
LOG_FILENAME = "dispnet"
CONFIG_FILENAME = "config.json"
debug = False
