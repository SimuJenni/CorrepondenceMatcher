import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

import os

from utils import montage_tf, get_variables_to_train
from constants import LOG_DIR

slim = tf.contrib.slim


class CMNetTrainer:
    def __init__(self, model, dataset, pre_processor, num_epochs, num_roi=4, optimizer='adam', lr_policy='const',
                 init_lr=0.0002, tag='default'):
        tf.logging.set_verbosity(tf.logging.DEBUG)
        self.sess = tf.Session()
        self.graph = tf.Graph()
        self.model = model
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.num_roi = num_roi
        self.tag = tag
        self.additional_info = None
        self.im_per_smry = 4
        self.summaries = {}
        self.pre_processor = pre_processor
        self.opt_type = optimizer
        self.lr_policy = lr_policy
        self.init_lr = init_lr
        self.num_train_steps = None
        with self.sess.as_default():
            with self.graph.as_default():
                self.global_step = slim.create_global_step()

    def get_save_dir(self):
        fname = '{}_{}_{}'.format(self.dataset.name, self.model.name, self.tag)
        if self.additional_info:
            fname = '{}_{}'.format(fname, self.additional_info)
        return os.path.join(LOG_DIR, '{}/'.format(fname))

    def optimizer(self):
        opts = {'adam': tf.train.AdamOptimizer(learning_rate=self.learning_rate(), beta1=0.5, epsilon=1e-6),
                'sgd+momentum': tf.train.MomentumOptimizer(learning_rate=self.learning_rate(), momentum=0.9)}
        return opts[self.opt_type]

    def learning_rate(self):
        policies = {'const': self.init_lr,
                    'linear': self.learning_rate_linear(self.init_lr)}
        return policies[self.lr_policy]

    def get_train_batch(self):
        with tf.device('/cpu:0'):
            # Get the training dataset
            train_set = self.dataset.get_trainset()
            self.num_train_steps = (self.dataset.get_num_train() / self.model.batch_size) * self.num_epochs
            print('Number of training steps: {}'.format(self.num_train_steps))
            provider = slim.dataset_data_provider.DatasetDataProvider(train_set, num_readers=2,
                                                                      common_queue_capacity=4 * self.model.batch_size,
                                                                      common_queue_min=self.model.batch_size*2)
            [img1] = provider.get(['image'])
            coords1 = self.create_train_coordinates(img1)

            img2 = tf.copy(img1)
            coords2 = tf.copy(coords1)

            # Preprocess data
            img1, coords1 = self.pre_processor.process_train(img1, coords1)
            img2, coords2 = self.pre_processor.process_train(img2, coords2)

            # Make batches
            imgs1, imgs2, coords1, coords2 = tf.train.batch([img1, img2, coords1, coords2], batch_size=self.model.batch_size*2, num_threads=8,
                                          capacity=self.model.batch_size*2)

            return imgs1, imgs2, coords1, coords2

    def create_train_coordinates(self, img):
        """Creates roi coordinates for training..
        Args:
            img:
        """
        #TODO
        return [10, 10]

    def roi_prediction_loss(self, preds, rois, margin, scope):
        for i in range(self.num_roi):
            roi_pred = preds[i]
            roi_target = rois[i]
            tf.contrib.losses.mean_squared_error(roi_pred, roi_target, scope=scope)
            non_targets = rois.remove(roi_target)
            for non_target in non_targets:
                d = roi_pred - non_target
                d_square = tf.reduce_sum(tf.square(d), [1, 2, 3])
                d = tf.sqrt(d_square)
                d_m = margin - d
                d_trunc = tf.maximum(d_m, 0)
                tf.contrib.losses.mean_squared_error(d_trunc, tf.zeros_like(d_trunc), scope=scope)
        losses_roi = slim.losses.get_losses(scope)
        losses_roi += slim.losses.get_regularization_losses(scope)
        roi_total_loss = math_ops.add_n(losses_roi, name='total_{}'.format(scope))
        return roi_total_loss

    def make_train_op(self, loss, vars2train=None, scope=None):
        if scope:
            vars2train = get_variables_to_train(trainable_scopes=scope)
        train_op = slim.learning.create_train_op(loss, self.optimizer(), variables_to_train=vars2train,
                                                 global_step=self.global_step, summarize_gradients=False)
        return train_op

    def make_summaries(self):
        # Handle summaries
        for variable in slim.get_model_variables():
            tf.histogram_summary(variable.op.name, variable)
        tf.scalar_summary('learning rate', self.learning_rate())

    def make_image_summaries(self, imgs1, dec_im1, dec_ed1, imgs2, dec_im2, dec_ed2):
        tf.image_summary('imgs/imgs1', montage_tf(imgs1, 1, self.im_per_smry), max_images=1)
        tf.image_summary('imgs/dec_im1', montage_tf(dec_im1, 1, self.im_per_smry), max_images=1)
        tf.image_summary('imgs/dec_ed1', montage_tf(dec_ed1, 1, self.im_per_smry), max_images=1)
        tf.image_summary('imgs/imgs2', montage_tf(imgs2, 1, self.im_per_smry), max_images=1)
        tf.image_summary('imgs/dec_im2', montage_tf(dec_im2, 1, self.im_per_smry), max_images=1)
        tf.image_summary('imgs/dec_ed2', montage_tf(dec_ed2, 1, self.im_per_smry), max_images=1)

    def learning_rate_linear(self, init_lr=0.0002):
        return tf.train.polynomial_decay(init_lr, self.global_step, self.num_train_steps, end_learning_rate=0.0)

    def train(self):
        with self.sess.as_default():
            with self.graph.as_default():
                imgs1, imgs2, coords1, coords2 = self.get_train_batch()

                # Create the model
                preds1, preds2, rois1, rois2 = self.model.net(imgs1, imgs2, coords1, coords2, reuse=None, training=True)

                # Compute losses
                roi_pred_loss = self.roi_prediction_loss(preds1, preds2, rois1, rois2)

                # Handle dependencies with update_ops (batch-norm)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                if update_ops:
                    updates = tf.group(*update_ops)
                    roi_pred_loss = control_flow_ops.with_dependencies([updates], roi_pred_loss)

                # Make summaries
                self.make_summaries()
                # self.make_image_summaries(imgs1, dec_im1, dec_ed1, imgs2, dec_im2, dec_ed2)

                # Generator training operations
                train_op_roi_pred = self.make_train_op(roi_pred_loss, scope='encoder, roi_regressor')

                # Start training
                slim.learning.train(train_op_roi_pred, self.get_save_dir(),
                                    save_summaries_secs=600,
                                    save_interval_secs=3000,
                                    log_every_n_steps=100,
                                    number_of_steps=self.num_train_steps)
