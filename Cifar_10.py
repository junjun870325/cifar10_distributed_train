
"""Builds the CIFAR-10 network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
from six.moves import urllib
import tensorflow as tf
import Cifar_10_Input
import tensorflow.contrib.slim as slim

conv1_dropout_rate = 0.2
conv2_dropout_rate = 0.3
conv3_dropout_rate = 0.4

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = Cifar_10_Input.NUM_CLASSES

def inference(images, 
              is_training=True,
              dropout_keep_prob=0.8,
              num_class=NUM_CLASSES,
              reuse=tf.AUTO_REUSE):
    weight_decay = 0.004
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(), 
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return inference_in(images, is_training=is_training,
              dropout_keep_prob=dropout_keep_prob, num_class=num_class, reuse=reuse)

# 这里简单的写了几层神经网络，最高准确率达到0.99以上，读者可以拟合自己的神经网络
def inference_in(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        num_class=NUM_CLASSES,
                        reuse=tf.AUTO_REUSE,
                        scope='InceptionResnetV2'):
    
    with tf.variable_scope(scope, 'InceptionResnetV2', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
      
                # 15 x 15 x 32
                net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                                  scope='Conv2d_1a_3x3')
                
                # 13 x 13 x 32
                net = slim.conv2d(net, 32, 3, padding='VALID',
                                  scope='Conv2d_2a_3x3')
                
                # 13 x 13 x 64
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                
                # 6 x 6 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_3a_3x3')
                
                 # 6 x 6 x 80
                net = slim.conv2d(net, 80, 1, padding='VALID',
                                  scope='Conv2d_3b_1x1')
                
                # 4 x 4 x 192
                net = slim.conv2d(net, 192, 3, padding='VALID',
                                  scope='Conv2d_4a_3x3')
                
                # 2 x 2 x 192
                net = slim.max_pool2d(net, 2, stride=2, padding='VALID',
                                      scope='MaxPool_5a_3x3')
                
                net = tf.reshape(net, shape=[-1, 192 * 2 * 2])
                
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='Dropout')
                
                net = slim.fully_connected(net, num_class, activation_fn=None, reuse=False)
                
    return net
