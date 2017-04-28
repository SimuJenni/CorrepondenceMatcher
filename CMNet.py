import tensorflow as tf
import tensorflow.contrib.slim as slim
from layers import lrelu, merge
import numpy as np

DEFAULT_FILTER_DIMS = [64, 96, 160, 256, 256]
DEFAULT_KERNELS = [3, 1, 3, 1, 3]
DEFAULT_STRIDES = [2, 1, 2, 1, 2]


def cmnet_argscope(activation=lrelu, kernel_size=(3, 3), padding='SAME', training=True, center=True,
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
        self.sc_factor = 4

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

    def split_coordinates(self, coords, num_roi):
        print('Before split: {}'.format(coords.get_shape()))
        split = tf.split(1, num_roi, coords)
        print('After split: {}'.format(split[0].get_shape()))
        return split

    def coords2indices(self, coords):
        return tf.concat(1, [tf.range(self.batch_size), coords, tf.range(DEFAULT_FILTER_DIMS[-1])])

    def predict_roi(self, context, reuse=None, training=True):
        with tf.variable_scope('roi_regressor', reuse=reuse):
            with slim.arg_scope(cmnet_argscope(padding='SAME', training=training)):
                net = slim.conv2d(context, num_outputs=512, stride=1, kernel_size=[3, 3], scope='conv_1')
                net = slim.conv2d(net, num_outputs=256, stride=1, kernel_size=[1, 1], scope='conv_1')
                return net

    def extract_roi(self, fmap, coord):
        print('Feature map: {}'.format(fmap.get_shape()))
        print('Feature map: {}'.format(tf.rank(fmap)))

        print('Coords: {}'.format(coord.get_shape()))
        coord = tf.squeeze(coord)
        print('Coords squeezed: {}'.format(coord.get_shape()))
        fmap_tr = tf.transpose(fmap, [0, 3, 1, 2])
        print('Feature map perm: {}'.format(fmap_tr.get_shape()))
        coord_tiled = tf.reshape(tf.tile(coord, [1, DEFAULT_FILTER_DIMS[-1]]),
                                 (self.batch_size, DEFAULT_FILTER_DIMS[-1], 2))
        print('Coord tiled: {}'.format(coord_tiled.get_shape()))
        roi = tf.gather_nd(fmap_tr, coord_tiled)
        print('Roi: {}'.format(roi.get_shape()))
        return roi

    def roi_context(self, fmap, coord):
        context_all = tf.slice(fmap, [coord[0]-1, coord[1]-1, 0], [3, 3, DEFAULT_FILTER_DIMS[-1]])
        zero_context = tf.zeros_like(context_all)
        mask = np.ones([self.batch_size, 3, 3, DEFAULT_FILTER_DIMS[-1]], dtype=bool)
        mask[:, 1, 1, :] = False
        context = tf.select(mask, context_all, zero_context)
        return context

    def roi_classifier(self, roi1, roi2, reuse=None, training=True):
        net = merge(roi1, roi2, dim=3)
        with tf.variable_scope('roi_classifier', reuse=reuse):
            with slim.arg_scope(cmnet_argscope(padding='SAME', training=training, center=True)):
                net = slim.conv2d(net, num_outputs=512, stride=1, kernel_size=[1, 1], scope='conv_1')
                net = slim.conv2d(net, num_outputs=256, stride=1, kernel_size=[1, 1], scope='conv_1')
                net = slim.conv2d(net, num_outputs=1, stride=1, kernel_size=[1, 1], scope='conv_1',
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
                for l in range(0, self.num_layers):
                    net = slim.conv2d(net, num_outputs=f_dims[l], stride=strides[l],
                                      kernel_size=[k_sizes[l], k_sizes[l]], scope='conv_{}'.format(l + 1))
                return net
