
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import math

from six.moves import urllib
import tensorflow as tf

import input

FLAGS = tf.app.flags.FLAGS

DATA_DIR = ''
use_fp16 = False
DROP_OUT = 0.5
MOMENTUM_USE = False
BATCH_NORM = True
batch_norm_decay = 0.9998
CONV_KERNEL_SIZE = 3 # originally 5
CONV_STRIDE = 1

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9998     # The decay to use for the moving average. originally 0.9999
L2_TERM_WEIGHT = 0.00009 # 0.0001
#FCN_WEIGHT_STDDEV = 0.02

FCN_NODE_NUM = 4096
BATCH_SIZE =1
NUM_CLASSES = 21  # change NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN accordingly
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
ROW_SIZE = 112
COL_SIZE = 112
FRAME_NUM = 25

LEARNING_RATE_DECAY_FACTOR = 0.005  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.09      # Initial learning rate. 

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _variable_on_cpu(name, shape, initializer):
  var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)    
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  var = _variable_on_cpu(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def batch_norm(inputs, n_out, k): 
  
  scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
  beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
  pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
  pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

  decay = batch_norm_decay
  epsilon = 1e-3
  return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)
  
def conv2d(input_data, conv, size, stride):
  with tf.variable_scope('%s'%(conv), reuse=tf.AUTO_REUSE) as scope:
    kernel = _variable_with_weight_decay('weights',shape=[CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, size[0], size[1]], stddev=5e-2, wd=0.0)
    conv1 = tf.nn.conv2d(input_data, kernel, [1, stride, stride, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [size[1]], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv1, biases)
    conv1_bn = batch_norm(pre_activation, size[1], 1)
    return tf.nn.leaky_relu(conv1_bn, alpha=0.1, name=scope.name)

def pool(input_name, output_name):
  return tf.nn.max_pool(input_name, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='%s'%(output_name))

def local2d(input_data, local):
  with tf.variable_scope('%s'%(local), reuse=tf.AUTO_REUSE) as scope:
    dim = 1
    for d in input_data.get_shape()[1:].as_list():
        dim *= d
    reshape = tf.reshape(input_data, [-1, dim])
    weights = _variable_with_weight_decay('weights', shape=[dim, FCN_NODE_NUM], stddev=1/math.sqrt(dim), wd=L2_TERM_WEIGHT)
    biases = _variable_on_cpu('biases', [FCN_NODE_NUM], tf.constant_initializer(0.0))
    local_matmul = tf.matmul(reshape, weights) + biases  
    local_bn = batch_norm(local_matmul, FCN_NODE_NUM, 0)
    return tf.nn.leaky_relu(local_bn, alpha=0.1, name=scope.name)


def inference(images):

  split = tf.split(images,FRAME_NUM,1) # TensorFlow 0.12 
  kernel_conv1 = _variable_with_weight_decay('weights', shape=[3, 3, 3, 48], stddev=5e-2, wd=0.0)
  biases_conv1 = _variable_on_cpu('biases_conv1', [48], tf.constant_initializer(0.0))
  pool_conv1 = [None] * 25
  
  for i in range(0, 25):
    split_tmp = tf.reshape(split[i], [BATCH_SIZE, ROW_SIZE, COL_SIZE, 3])
    split_tmp = tf.cast(split_tmp, tf.float32)
    conv_0 = tf.nn.conv2d(split_tmp, kernel_conv1, [1, 1, 1, 1], padding='SAME')
    pre_activation_0 = tf.nn.bias_add(conv_0, biases_conv1)
    conv_relu = tf.nn.leaky_relu(pre_activation_0, alpha=0.1)    
    pool_conv1[i] = tf.nn.max_pool(conv_relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
  
  pool_aggregated = tf.concat([pool_conv1[0], pool_conv1[1], pool_conv1[2], pool_conv1[3], pool_conv1[4], pool_conv1[5], 
                                pool_conv1[6], pool_conv1[7], pool_conv1[8], pool_conv1[9], pool_conv1[10], pool_conv1[11], 
                                pool_conv1[12], pool_conv1[13], pool_conv1[14], pool_conv1[15], pool_conv1[16], pool_conv1[17], 
                                pool_conv1[18], pool_conv1[19], pool_conv1[20], pool_conv1[21], pool_conv1[22], pool_conv1[23], pool_conv1[24]],3)
    
  # conv dimension reduction
  with tf.variable_scope('conv_reduction',reuse=tf.AUTO_REUSE) as scope:
    kernel_reduction = _variable_with_weight_decay('weights',shape=[1, 1, 1200, 96],stddev=5e-2,wd=0.0)
    conv = tf.nn.conv2d(pool_aggregated, kernel_reduction, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_reduction = tf.nn.leaky_relu(pre_activation, alpha=0.1, name=scope.name)
  
  conv1 = conv2d(conv_reduction, 'conv1', [96, 256], 2)
  pool1 = pool(conv1, 'pool1')
  conv2 = conv2d(pool1, 'conv2', [256, 512], 2)
  conv3 = conv2d(conv2, 'conv3', [512, 512], CONV_STRIDE)
  conv4 = conv2d(conv3, 'conv4', [512, 512], CONV_STRIDE)
  pool2 = pool(conv4, 'pool2')
  local3 = local2d(pool2, 'local3')
  local4 = local2d(local3, 'local4')
  
  with tf.variable_scope('softmax_linear',reuse=tf.AUTO_REUSE) as scope:
    dim = 1
    for d in local4.get_shape()[1:].as_list():
        dim *= d
    #print('dim : ', dim)
    weights = _variable_with_weight_decay('weights', [FCN_NODE_NUM, NUM_CLASSES], stddev=1/math.sqrt(dim), wd=L2_TERM_WEIGHT)
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name) 

  return softmax_linear