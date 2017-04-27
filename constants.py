import os

NUM_THREADS = 16

# Directories for data, images and models
DATA_DIR = '/data/cvg/simon/data/'
LOG_DIR = os.path.join(DATA_DIR, 'logs_CMatch/')

# Directories for tf-records
OBJECTNET3D_TF_DATADIR = os.path.join(DATA_DIR, 'ObjectNet3D-TFRecords/')
IMAGENET_TF_256_DATADIR = os.path.join(DATA_DIR, 'imagenet-TFRecords/')

# Source directories for datasets
OBJECTNET3D_DATADIR = os.path.join(DATA_DIR, 'ObjectNet3D/')