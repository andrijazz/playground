# import os
#
# # check if $DATA_DRIVE env var is set
# DATA_DRIVE = os.getenv('DATA_DRIVE')
# if not DATA_DRIVE:
#     exit("DATA_DRIVE env variable is not set. "
#          "Please expand ~/.profile file with following command: "
#          "export DATA_DRIVE=<your_drive_dir>")
#
# # this is abs path of directory of this file
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#
# # root dir is always parent dir to model files
# ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
#
# # check if project root folder is in $PYTHONPATH
# python_path_string = os.getenv('PYTHONPATH')
# if ROOT_DIR not in python_path_string.split(':'):
#     exit("ROOT dir of the playground project is not in PYTHONPATHs")
#
# # root folder for all the datasets is DATASET_DIR (for ex. /mnt/datasets)
# DATASET_DIR = os.path.join(DATA_DRIVE, "datasets")
# MODEL_NAME = "fcn"
# LOG_FILENAME = "fcn"
# CONFIG_FILENAME = "config.json"
#
# # log directory of the model is LOG_DIR (for ex. /mnt/log/fcn)
# LOG_DIR = os.path.join(DATA_DRIVE, "log", MODEL_NAME)
#
