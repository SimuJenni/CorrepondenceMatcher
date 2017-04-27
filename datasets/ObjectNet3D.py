import tensorflow as tf
import os

from Dataset import Dataset
from dataset_utils import load_obj

slim = tf.contrib.slim
from constants import OBJECTNET3D_TF_DATADIR


class ObjectNet3D(Dataset):

    SPLITS_TO_SIZES = {'train': 2501, 'val': 2510, 'trainval': 5011, 'test': 4952}

    ITEMS_TO_DESCRIPTIONS = {
        'image': 'A color image.',
        'label': 'A single integer between 0 and 19 or -1 for unlabeled',
        'im_size': '(h,w) indicating the size of the image',
        'bbox': 'A list of bounding box coordinates',
        'viewpoint': 'A list of viewpoint coordinates [azimuth, elevation, theta]'
    }

    def __init__(self, class_name):
        Dataset.__init__(self)
        self.data_dir = OBJECTNET3D_TF_DATADIR
        self.class_name = class_name
        self.num_classes = 100
        self.name = 'ObjectNet3D'
        self.is_multilabel = True

    def get_data_files(self, split_name):
        tf_record_pattern = os.path.join(self.data_dir, '%s/ON3D_%s.tfrecord' % (split_name, self.class_name))
        data_files = tf.gfile.Glob(tf_record_pattern)
        return data_files

    def get_keys_to_features(self):
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
            'image/im_size': tf.FixedLenFeature([3], tf.int64),
            'image/bbox': tf.FixedLenFeature([4], tf.float32),
            'image/viewpoint': tf.FixedLenFeature([3], tf.float32),
        }
        return keys_to_features

    def get_items_to_handlers(self):
        items_to_handlers = {
            'image': slim.tfexample_decoder.Image('image/encoded', 'image/format', channels=3),
            'im_size': slim.tfexample_decoder.Tensor('image/im_size'),
            'bbox': slim.tfexample_decoder.Tensor('image/bbox'),
            'viewpoint': slim.tfexample_decoder.Tensor('image/viewpoint')
        }
        return items_to_handlers

    def format_labels(self, labels):
        return labels

    def get_trainset(self):
        return self.get_split('trainval')

    def get_testset(self):
        return self.get_split('test')

    def get_num_dataset(self, split_name):
        dict_dir = os.path.join(self.data_dir, split_name)
        num_per_class = load_obj('num_per_class', dict_dir)
        return num_per_class[self.class_name]

    def get_num_test(self):
        return self.get_num_dataset('test')

    def get_num_train(self):
        return self.get_num_dataset('trainval')
