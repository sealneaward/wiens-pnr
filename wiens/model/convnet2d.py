from __future__ import division
import tensorflow as tf
import numpy as np
from pnr.model.utils import *


class ConvNet2d:
    """

    """

    def __init__(self, config):
        self.config = config
        self.d1 = config['d1']
        self.d2 = config['d2']
        self.conv_layers = config['conv_layers']
        self.fc_layers = config['fc_layers']
        self.value_keep_prob = config['keep_prob']
        self.activation = config['activation']
        if 'bn' in config:## later version
            self.bn = config['bn']
            if self.bn:
                self.training = tf.placeholder(tf.bool)
                self.bn_inds = config['bn_inds']
        else:
            self.bn = False
        if 'pool' in config:## later version
            self.pool = config['pool']
        else:
            self.pool = False
    # convnet

    def build(self):
        # placeholders
        x = tf.placeholder(tf.float32, [None, self.d1, self.d2, 3])
        keep_prob = tf.placeholder(tf.float32)

        # init weights/bias
        # conv
        W_conv = []
        b_conv = []
        for layer_ind in xrange(len(self.conv_layers)):
            W_conv.append(weight_variable(self.conv_layers[layer_ind]))
            b_conv.append(bias_variable([self.conv_layers[layer_ind][-1]]))
        if self.pool:
            SHAPE_convlast = int(np.ceil(self.d1 / (2**len(self.conv_layers))) *
                                 np.ceil(self.d2 / (2**len(self.conv_layers))) *
                                 self.conv_layers[-1][-1])
        else:
            SHAPE_convlast = int(np.ceil(self.d1 ) *
                                 np.ceil(self.d2 ) *
                                 self.conv_layers[-1][-1])
        # fc
        W_fc = []
        b_fc = []
        self.fc_layers.insert(0, SHAPE_convlast)
        for layer_ind in xrange(len(self.fc_layers) - 1):
            W_fc.append(weight_variable(
                [self.fc_layers[layer_ind], self.fc_layers[layer_ind + 1]]))
            b_fc.append(bias_variable([self.fc_layers[layer_ind + 1]]))

        # build model
        # conv
        h_pool_drop = x
        for layer_ind in xrange(len(self.conv_layers)):
            h_conv = tf.nn.relu(
                conv2d(h_pool_drop, W_conv[layer_ind]) + b_conv[layer_ind]
            )
            if self.pool:
                h_pool = max_pool_2x2(h_conv)
            else:
                h_pool = h_conv
            if self.bn and layer_ind in self.bn_inds:
                h_pool = bn(h_pool, self.training)
            h_pool_drop = tf.nn.dropout(h_pool, keep_prob)
        # fc
        h_pool_flat = tf.reshape(h_pool_drop, [-1, SHAPE_convlast])
        h_fc_drop = h_pool_flat
        for layer_ind in xrange(len(self.fc_layers) - 2):
            h_fc = tf.nn.relu(
                tf.matmul(h_fc_drop, W_fc[layer_ind]) + b_fc[layer_ind]
            )
            if self.bn:
                h_fc = bn(h_fc, self.training)
            h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

        y = tf.matmul(h_fc_drop, W_fc[-1]) + b_fc[-1]

        self.x = x
        self.keep_prob = keep_prob
        self.logits = y

    def input(self, x, keep_prob=None, training=False):
        if keep_prob == None: #default, 'training'
            keep_prob = self.value_keep_prob
        ret_dict = {}
        ret_dict[self.x] = x
        ret_dict[self.keep_prob] = keep_prob
        if self.bn:
            ret_dict[self.training] = training
        return ret_dict

    def output(self):
        return self.logits