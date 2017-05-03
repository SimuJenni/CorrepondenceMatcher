import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

import os
import numpy as np

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
        self.num_eval_steps = None
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
            rel_coords1 = self.sample_coordinates()

            img2 = tf.identity(img1)
            rel_coords2 = tf.identity(rel_coords1)

            # Preprocess data
            img1, im_coords1 = self.pre_processor.process_train(img1, rel_coords1, 0)
            img2, im_coords2 = self.pre_processor.process_train(img2, rel_coords2, 1)
            roi_coords1 = self.model.im2roi_coords(im_coords1)
            roi_coords2 = self.model.im2roi_coords(im_coords2)

            # Make batches
            imgs1, imgs2, roi_coords1, roi_coords2 = tf.train.batch([img1, img2, roi_coords1, roi_coords2],
                                                            batch_size=self.model.batch_size,
                                                            num_threads=8,
                                                            capacity=self.model.batch_size)
            return imgs1, imgs2, roi_coords1, roi_coords2

    def get_test_batch(self, num_eval=None):
        with tf.device('/cpu:0'):
            # Get the training dataset
            test_set = self.dataset.get_testset()
            if num_eval:
                self.num_eval_steps = num_eval
            else:
                self.num_eval_steps = (self.dataset.get_num_test() / self.model.batch_size)
            provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=1, shuffle=False)

            [img1] = provider.get(['image'])
            coords1 = self.sample_coordinates()
            img2 = tf.identity(img1)
            coords2 = tf.identity(coords1)

            # Preprocess data
            img1, coords1 = self.pre_processor.process_test(img1, coords1, 0)
            img2, coords2 = self.pre_processor.process_test(img2, coords2, 1)

            # Make batches
            imgs1, imgs2, coords1, coords2 = tf.train.batch([img1, img2, coords1, coords2],
                                                            batch_size=self.model.batch_size,
                                                            num_threads=1)
            return imgs1, imgs2, coords1, coords2

    def sample_coordinates(self):
        """Creates roi coordinates for training. Coordinates are relative to image shape
        (i.e., values in the interval [0,1])

        Args:
            img:
        """
        shape = (self.num_roi, 2)
        return tf.random_uniform(shape=shape, minval=0.1, maxval=0.9)

    def roi_prediction_loss(self, preds, rois, margin, scope):
        for i in range(self.num_roi):
            roi_pred = preds[i]
            roi_target = rois[i]
            tf.contrib.losses.mean_squared_error(predictions=roi_pred, labels=roi_target, scope=scope)
            non_targets = rois[:i] + rois[i+1:]
            for non_target in non_targets:
                d = roi_pred - non_target
                d_square = tf.reduce_sum(tf.square(d), [1, 2, 3])
                d = tf.sqrt(d_square)
                d_m = margin - d
                d_trunc = tf.maximum(d_m, 0)
                tf.contrib.losses.mean_squared_error(predictions=d_trunc, labels=tf.zeros_like(d_trunc), scope=scope)
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
                preds1, preds2, rois1, rois2 = self.model.net(imgs1, imgs2, coords1, coords2, num_roi=self.num_roi,
                                                              reuse=None, training=True)

                # Compute losses
                margin = 10
                scope = 'roi_pred_loss'
                roi_pred_loss = self.roi_prediction_loss(preds1, rois1, margin=margin, scope=scope)
                roi_pred_loss += self.roi_prediction_loss(preds2, rois2, margin=margin, scope=scope)
                tf.scalar_summary('losses/roi_pred_loss', roi_pred_loss)

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

    def test(self, num_eval=None):
        with self.sess.as_default():
            with self.graph.as_default():
                imgs1, imgs2, coords1, coords2 = self.get_test_batch(num_eval)

                # Create the model
                rois1, pred2, roi1 = self.model.predict(imgs1, coords1, self.num_roi, reuse=None)
                rois2, pred1, roi2 = self.model.predict(imgs2, coords2, self.num_roi, reuse=True)

                # Make summaries
                dist_img1 = tf.reduce_mean(tf.square(rois1-pred1[0]), axis=3, keep_dims=True)
                dist_img2 = tf.reduce_mean(tf.square(rois2-pred2[0]), axis=3, keep_dims=True)

                names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
                    'mse1': slim.metrics.streaming_mean_squared_error(pred1[0], roi1[0]),
                    'mse2': slim.metrics.streaming_mean_squared_error(pred2[0], roi2[0]),
                })

                summary_ops = []
                for metric_name, metric_value in names_to_values.iteritems():
                    op = tf.scalar_summary(metric_name, metric_value)
                    op = tf.Print(op, [metric_value], metric_name)
                    summary_ops.append(op)

                summary_ops.append(tf.image_summary('imgs/dist_img1', montage_tf(dist_img1, 1, self.im_per_smry),
                                                    max_images=1))
                summary_ops.append(tf.image_summary('imgs/dist_img2', montage_tf(dist_img2, 1, self.im_per_smry),
                                                    max_images=1))
                summary_ops.append(tf.image_summary('imgs/imgs1', montage_tf(imgs1, 1, self.im_per_smry),
                                                    max_images=1))
                summary_ops.append(tf.image_summary('imgs/imgs2', montage_tf(imgs2, 1, self.im_per_smry),
                                                    max_images=1))

                # Start evaluation
                slim.evaluation.evaluation_loop('', self.get_save_dir(), self.get_save_dir(),
                                                num_evals=self.num_eval_steps,
                                                max_number_of_evaluations=1,
                                                eval_op=names_to_updates.values(),
                                                summary_op=tf.merge_summary(summary_ops))


