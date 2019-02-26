import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
sys.path.insert(0, ROOT_DIR)

MODEL_NAME = "fcn"
LOG_FILENAME = "fcn"
CONFIG_FILENAME = "config.json"
DATA_DRIVE = "/datadrive"

DATASET_DIR = DATA_DRIVE + "/datasets"
LOG_DIR = DATA_DRIVE + "/log/" + MODEL_NAME
debug = False

