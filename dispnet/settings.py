import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

MODEL_NAME = "dispnet"
LOG_FILENAME = "dispnet"
CONFIG_FILENAME = "config.json"
# DATA_DRIVE = "/datadrive"
DATA_DRIVE = "/mnt"

DATASET_DIR = DATA_DRIVE + "/datasets"
LOG_DIR = DATA_DRIVE + "/log/" + MODEL_NAME
debug = False

