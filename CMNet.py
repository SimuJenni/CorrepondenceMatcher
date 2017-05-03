import tensorflow as tf
import tensorflow.contrib.slim as slim
from layers import merge
import numpy as np

DEFAULT_FILTER_DIMS = [64, 96, 160, 256, 256]
DEFAULT_KERNELS = [3, 3, 3, 3, 3]
DEFAULT_STRIDES = [2, 1, 2, 1, 2]


def cmnet_argscope(activation=tf.nn.relu, kernel_size=(3, 3), padding='SAME', training=True, center=True,
                   w_reg=0.0001, fix_bn=False):
    """Defines default parameter values for all the layers used in ToonNet.

    Args:
        activation: The default activation function
        kernel_size: The default kernel size for convolution layers
        padding: The default border mode
        training: Whether in train or test mode
        center: Whether to use centering in batchnorm
        w_reg: Parameter for weight-decay

    Returns:
        An argscope
    """
    train_bn = training and not fix_bn
    batch_norm_params = {
        'is_training': train_bn,
        'decay': 0.95,
        'epsilon': 0.001,
        'center': center,
    }
    trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.convolution2d_transpose],
                        activation_fn=activation,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(w_reg),
                        biases_initializer=tf.constant_initializer(0.1)):
        with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose],
                            kernel_size=kernel_size,
                            padding=padding):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.dropout], is_training=training) as arg_sc:
                    with slim.arg_scope([slim.fully_connected],
                                        weights_initializer=trunc_normal(0.005)):
                        return arg_sc


class CMNet:
    def __init__(self, num_layers, batch_size, tag='default'):
        """Initialises a VPNet using the provided parameters.

        Args:
            num_layers: The number of convolutional down/upsampling layers to be used.
            batch_size: The batch-size used during training (used to generate training labels)
            vgg_discriminator: Whether to use VGG-A instead of AlexNet in the discriminator
        """
        self.name = 'CMNet_{}'.format(tag)
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.sc_factor = np.prod(DEFAULT_STRIDES)

    def net(self, im1, im2, coords1, coords2, num_roi, reuse=None, training=True):
        """

        Args:
            im1:
            im2:
            coords1: Tensor of ROI coordinates in im1
            coords2: Tensor of ROI coordinates correpsonding to coords1 in im2
            reuse:
            training:
        """
        rois1 = []
        rois2 = []
        preds1 = []
        preds2 = []

        cs1 = self.split_coordinates(coords1, num_roi)
        cs2 = self.split_coordinates(coords2, num_roi)

        enc1 = self.encoder(im1, reuse=reuse, training=training)
        enc2 = self.encoder(im2, reuse=True, training=training)
        for i, (c1, c2) in enumerate(zip(cs1, cs2)):
            idx1 = self.coords2indices(c1)
            idx2 = self.coords2indices(c2)
            rois1.append(self.extract_roi(enc1, idx1))
            rois2.append(self.extract_roi(enc2, idx2))
            context1 = self.roi_context(enc1, idx1)
            context2 = self.roi_context(enc2, idx1)
            preds1.append(self.predict_roi(context1, reuse=reuse if i == 0 else True, training=training))
            preds2.append(self.predict_roi(context2, reuse=True, training=training))
        return preds1, preds2, rois1, rois2

    def predict(self, imgs, coords, num_roi, reuse=None, training=False):
        rois = []
        preds = []

        enc = self.encoder(imgs, reuse=reuse, training=training)
        cs = self.split_coordinates(coords, num_roi)

        for i, c in enumerate(cs):
            idx = self.coords2indices(c)
            rois.append(self.extract_roi(enc, idx))
            context = self.roi_context(enc, idx)
            preds.append(self.predict_roi(context, reuse=reuse if i == 0 else True, training=training))
        return enc, preds, rois

    def im2roi_coords(self, coords):
        return tf.round(coords/self.sc_factor)

    def split_coordinates(self, coords, num_roi):
        split = tf.split(1, num_roi, coords)
        return split

    def coords2indices(self, coords):
        return tf.concat(1, [tf.reshape(tf.range(self.batch_size), [self.batch_size, 1]), tf.squeeze(coords)])

    def idx2context(self, idx):
        context_inds = []
        context_offsets = [[0, -1, -1], [0, 0, -1], [0, 1, -1], [0, -1, 0], [0, 1, 0], [0, -1, 1], [0, 0, 1], [0, 1, 1]]
        for offset in context_offsets:
            context_inds.append(idx + offset)

        return context_inds

    def predict_roi(self, context, reuse=None, training=True):
        with tf.variable_scope('roi_regressor', reuse=reuse):
            with slim.arg_scope(cmnet_argscope(padding='SAME', training=training)):
                net = slim.conv2d(context, num_outputs=512, stride=1, kernel_size=[3, 3], scope='conv_1')
                net = slim.conv2d(net, num_outputs=256, stride=1, kernel_size=[1, 1], scope='conv_2')
                net = slim.conv2d(net, num_outputs=256, stride=1, kernel_size=[1, 1], scope='conv_3',
                                  normalizer_fn=None)
                return net

    def extract_roi(self, fmap, coord):
        roi = tf.gather_nd(fmap, coord, name='GatherND_extract_roi')
        roi = tf.reshape(roi, [self.batch_size, 1, 1, DEFAULT_FILTER_DIMS[-1]])
        return roi

    def roi_context(self, fmap, coord):
        context_inds = self.idx2context(coord)
        context_rois = [self.extract_roi(fmap, idx) for idx in context_inds]
        context = tf.concat(3, context_rois)
        return context

    def roi_classifier(self, roi1, roi2, reuse=None, training=True):
        net = merge(roi1, roi2, dim=3)
        with tf.variable_scope('roi_classifier', reuse=reuse):
            with slim.arg_scope(cmnet_argscope(padding='SAME', training=training, center=True)):
                net = slim.conv2d(net, num_outputs=512, stride=1, kernel_size=[1, 1], scope='conv_1')
                net = slim.conv2d(net, num_outputs=256, stride=1, kernel_size=[1, 1], scope='conv_2')
                net = slim.conv2d(net, num_outputs=1, stride=1, kernel_size=[1, 1], scope='conv_3',
                                  activation_fn=None, normalizer_fn=None)
                return net

    def encoder(self, net, reuse=None, training=True):
        """Builds an encoder of the given inputs.

        Args:
            net: Input to the encoder (image)
            reuse: Whether to reuse already defined variables
            training: Whether in train or test mode.

        Returns:
            Encoding of the input image.
        """
        f_dims = DEFAULT_FILTER_DIMS
        strides = DEFAULT_STRIDES
        k_sizes = DEFAULT_KERNELS
        with tf.variable_scope('encoder', reuse=reuse):
            with slim.arg_scope(cmnet_argscope(padding='SAME', training=training)):
                for l in range(0, self.num_layers-1):
                    net = slim.conv2d(net, num_outputs=f_dims[l], kernel_size=[k_sizes[l], k_sizes[l]],
                                      scope='conv_{}'.format(l + 1))
                    if strides[l] > 1:
                        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=strides[l], scope='pool_{}'.format(l))

                net = slim.conv2d(net, num_outputs=f_dims[-1],kernel_size=[k_sizes[-1], k_sizes[-1]],
                                  scope='conv_{}'.format(self.num_layers), normalizer_fn=None)
                if strides[-1] > 1:
                    net = slim.max_pool2d(net, kernel_size=[3, 3], stride=strides[-1],
                                          scope='pool_{}'.format(self.num_layers))
                return net
