from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import scipy.io

import xml.etree.ElementTree as ET
from scipy import misc

import numpy as np
import tensorflow as tf
from constants import OBJECTNET3D_DATADIR, OBJECTNET3D_TF_DATADIR

from datasets import dataset_utils

# The names of the classes.
NUM_CLASSES = 100
CLASSES = ['eyeglasses', 'laptop', 'skate', 'shoe', 'keyboard', 'chair', 'flashlight', 'cup', 'backpack', 'jar',
           'bench', 'piano', 'stove', 'plate', 'watch', 'trash_bin', 'spoon', 'fan', 'lighter', 'stapler', 'ashtray',
           'sofa', 'pan', 'bicycle', 'sign', 'computer', 'tvmonitor', 'racket', 'hammer', 'fire_extinguisher', 'clock',
           'kettle', 'microwave', 'pen', 'knife', 'iron', 'mailbox', 'guitar', 'key', 'washing_machine', 'tub', 'comb',
           'hair_dryer', 'trophy', 'toothbrush', 'faucet', 'skateboard', 'remote_control', 'pillow', 'shovel', 'pot',
           'headphone', 'slipper', 'scissors', 'filing_cabinet', 'boat', 'motorbike', 'camera', 'refrigerator',
           'eraser', 'fork', 'pencil', 'door', 'bus', 'dishwasher', 'cabinet', 'train', 'rifle', 'bookshelf', 'teapot',
           'car', 'cap', 'bucket', 'bed', 'can', 'aeroplane', 'suitcase', 'toilet', 'fish_tank', 'calculator',
           'telephone', 'satellite_dish', 'mouse', 'helmet', 'microphone', 'toaster', 'speaker', 'paintbrush',
           'printer', 'vending_machine', 'blackboard', 'coffee_maker', 'screwdriver', 'road_pole',
           'diningtable', 'cellphone', 'wheelchair', 'desk_lamp', 'bottle', 'basket']


def parse_mat(mat_file, data_path):
    """Parse .mat file with viewpoint annotations
    Args:
      mat_file: the input mat file path
    Returns:
      examples: list of example objects with annotations
    """
    examples = []

    mat = scipy.io.loadmat(mat_file)
    record = mat['record']

    # Get the filename
    fname = record['filename'][0, 0][0]
    image_path = os.path.join(data_path, 'Images/ObjectNet3D/Images', fname)

    # Get the image size
    im_size = record['imgsize'][0, 0][0]

    # Get all the objects
    objects = mat['record']['objects'][0, 0]
    num_objects = objects.shape[1]
    for i in range(num_objects):
        class_name = objects[0, i]['class'][0]
        bbox = objects[0, i]['bbox'][0]

        # Get the viewpoint information
        viewpoint = objects[0, i]['viewpoint'][0, 0]
        azimuth = viewpoint['azimuth_coarse'][0, 0]
        elevation = viewpoint['elevation_coarse'][0, 0]
        theta = viewpoint['theta'][0, 0]

        difficult = objects[0, i]['difficult'][0, 0]
        occluded = objects[0, i]['occluded'][0, 0]
        truncated = objects[0, i]['truncated'][0, 0]

        if difficult+occluded+truncated == 0:
            examples.append({'image_path': image_path, 'im_size': im_size, 'class_name': class_name, 'bbox': bbox,
                            'azimuth': azimuth, 'elevation': elevation, 'theta': theta})
    return examples


def to_tfrecord(image_ids_file, dest_dir, source_dir):
    if not tf.gfile.Exists(dest_dir):
        tf.gfile.MakeDirs(dest_dir)
    with open(image_ids_file) as f:
        img_ids = f.readlines()

    num_images = len(img_ids)
    num_per_class = {cn: 0 for cn in CLASSES}
    writers = {c: tf.python_io.TFRecordWriter(get_output_filename(dest_dir, c)) for c in CLASSES}

    with tf.Graph().as_default():
        coder = dataset_utils.ImageCoder()

        with tf.Session('') as sess:
            for j in range(num_images):
                # Parse the annotations file
                mat_path = os.path.join(source_dir, 'Annotations', '{}.mat'.format(img_ids[j].strip('\n')))
                examples = parse_mat(mat_path, source_dir)

                for e in examples:
                    sys.stdout.write('\r>> Reading file [%s] image %d/%d' % (e['image_path'], j + 1, num_images))
                    sys.stdout.flush()

                    # Get image, edge-map and cartooned image
                    img = misc.imread(e['image_path'], mode='RGB')

                    # Encode the images
                    image_str = coder.encode_jpeg(img)

                    # Build example
                    example = dataset_utils.to_tfexample(image_str, 'jpg', e['im_size'].tolist(), e['bbox'].tolist(),
                                                         e['azimuth'], e['elevation'], e['theta'])
                    # Write example
                    writers[e['class_name']].write(example.SerializeToString())

                    # Update number of examples per class
                    num_per_class[e['class_name']] += 1

    print(num_per_class)
    dataset_utils.save_obj(num_per_class, dest_dir, 'num_per_class')


def get_output_filename(dest_dir, class_name):
    """Creates the output filename.
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      split_name: The name of the train/test split.
    Returns:
      An absolute file path.
    """
    return '%s/ON3D_%s.tfrecord' % (dest_dir, class_name)


def run(target_dir=OBJECTNET3D_TF_DATADIR, source_dir=OBJECTNET3D_DATADIR):
    """Runs the conversion operation.
    Args:
      target_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(target_dir):
        tf.gfile.MakeDirs(target_dir)

    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    trainval_dir = os.path.join(target_dir, 'trainval')
    testing_dir = os.path.join(target_dir, 'test')

    # First, process the trainval data:
    filename = os.path.join(source_dir, 'Image_sets', 'trainval.txt')
    to_tfrecord(filename, trainval_dir, source_dir)

    # # Process the train data:
    # filename = os.path.join(source_dir, 'Image_sets', 'train.txt')
    # to_tfrecord(filename, train_dir, source_dir)
    #
    # # Process the val data:
    # filename = os.path.join(source_dir, 'Image_sets', 'val.txt')
    # to_tfrecord(filename, val_dir, source_dir)

    # Process the test data:
    filename = os.path.join(source_dir, 'Image_sets', 'test.txt')
    to_tfrecord(filename, testing_dir, source_dir)

    print('\nFinished converting the ObjectNet3D dataset!')


if __name__ == '__main__':
    run()
