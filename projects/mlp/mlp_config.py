from easydict import EasyDict as edict

C = edict()

# Alias
cfg = C

# Net name
C.NET = "MLPNet"

# Model to use
C.MODEL = "MLPModel"

# GPU to use
C.GPU = 0

# conf
# C.ARCH = [28 * 28, 1000, 1000, 500, 200, 10]
C.ARCH = [111, 64, 64, 8]
C.ACTIVATION = 'tanh'
C.ADD_OUTPUT_LAYER_ACTIVATION = True

########################################################################################################################
# Train options
########################################################################################################################

# Dataset for training
C.TRAIN_DATASET = "MNIST"
# Dataset augmentations
C.TRAIN_DATASET_AUG = []
# Step period to write summaries to tensorboard
C.TRAIN_SUMMARY_FREQ = 500
# Step period to perform validation
C.TRAIN_VAL_FREQ = 1000
C.TRAIN_START_SAVING_AFTER_EPOCH = 1
# Train batch size
C.TRAIN_BATCH_SIZE = 20
# Number of epochs
C.TRAIN_NUM_EPOCHS = 3
# Restore from wandb or local
C.TRAIN_RESTORE_STORAGE = 'local'
# Path to checkpoint from which to be restored
C.TRAIN_RESTORE_FILE = ''
# Learning rate
C.TRAIN_LR = 1e-3
# save checkpoint model frequency
C.TRAIN_SAVE_MODEL_FREQ = 5000
# L1
C.TRAIN_L1 = 0

########################################################################################################################
# Validation options.
########################################################################################################################

# Validation batch size
C.VAL_BATCH_SIZE = 20

########################################################################################################################
# Test options
########################################################################################################################

# Test batch size
C.TEST_BATCH_SIZE = 20
# Path to checkpoint from which to be restored
C.TEST_RESTORE_FILE = ""
# Restore from wandb or local
C.TEST_RESTORE_STORAGE = ""
# Dataset for testing
C.TEST_DATASET = "MNIST"
# Dataset augmentations
C.TEST_DATASET_AUG = []
