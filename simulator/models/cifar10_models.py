from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import collections

from simulator.models.mnist_models import nn_layer, flatten
from simulator.graph.device_placer import _gpu_device_name
from simulator.simulator_utils import DTYPE
def get_shape(tensor):
  return tensor.get_shape().as_list()

def nn_cifar10_model2(graph):
  """Creates model for NN mnist.

  Returns:
    logits
  """
  n_inputs = 32*32
  n_hidden1 = 1024
  n_hidden2 = 1024
  n_hidden3 = 2048
  n_outputs = 10

  with graph.as_default():
    with tf.name_scope('Inputs'):
      with tf.name_scope('X'):
        X = tf.placeholder(DTYPE, shape=(None, n_inputs), name='X') # pylint: disable=invalid-name
      with tf.name_scope('y'):
        y = tf.placeholder(tf.int64, shape=(None), name='y') # pylint: disable=invalid-name

    hidden1 = nn_layer(
        X,
        n_hidden1,
        name='hidden1',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden2 = nn_layer(
        hidden1,
        n_hidden2,
        name='hidden2',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden3 = nn_layer(
        hidden2,
        n_hidden3,
        name='hidden2',
        activation=tf.nn.relu)# pylint: disable=no-member

    logits = nn_layer(
        hidden3,
        n_outputs,
        name='logits')

  return X, y, logits

def cnn_cifar10_model(graph):

  with graph.as_default():
    with tf.name_scope('Inputs'):
      with tf.name_scope('X'):
        X = tf.placeholder(DTYPE, shape=(None, 3072), name='X')
        X_reshaped = tf.reshape(X, shape=tf.TensorShape([-1, 32, 32, 3]))
      with tf.name_scope('y'):
        y = tf.placeholder(tf.int64, shape=(None), name='y')

    with tf.device(_gpu_device_name(0)):
      with tf.name_scope('conv1'):
        kernel = _variable_with_weight_decay( name='kernel1',
                            shape=[5, 5, 3, 64],
                            stddev=5e-2,
                            wd=None)
        conv = tf.nn.conv2d(input=X_reshaped,
                  filter=kernel,
                  strides=[1, 1, 1, 1],
                  padding='SAME')
        biases = tf.get_variable(name='biases1', shape=[64], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation)

      with tf.name_scope('pool1'):

        pool1 = tf.nn.max_pool( conv1, 
                    ksize=[1, 3, 3, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    name='pool1')

      with tf.name_scope('norm1'):
        norm1 = tf.nn.lrn(  pool1, 
                  4, 
                  bias=1.0, 
                  alpha=0.001 / 9.0, 
                  beta=0.75,
                  name='norm1')

      with tf.name_scope('conv2'):
        kernel = _variable_with_weight_decay( 'kernel2',
                            shape=[5, 5, 64, 64],
                            stddev=5e-2,
                            wd=None)
        conv = tf.nn.conv2d(input=norm1,
                  filter=kernel,
                  strides=[1, 1, 1, 1],
                  padding='SAME')
        biases = tf.get_variable(name='biases2', shape=[64], initializer=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation)

      with tf.name_scope('norm2'):
        norm2 = tf.nn.lrn(  conv2, 
                  4, 
                  bias=1.0, 
                  alpha=0.001 / 9.0, 
                  beta=0.75,
                  name='norm2')

      with tf.name_scope('pool2'):
        pool2 = tf.nn.max_pool( norm2, 
                    ksize=[1, 3, 3, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    name='pool2')
      


      with tf.name_scope('fully_connected1'):
        reshaped = tf.reshape(pool2, [X_reshaped.get_shape().as_list()[0], -1])
        fc1 = nn_layer(reshaped, 384, 'fully_connected1', tf.nn.relu)

      keep_prob = tf.placeholder(DTYPE, name='keep_prob')
      with tf.name_scope('dropout1'):
        fc1_dropout = tf.nn.dropout(fc1, keep_prob)

      with tf.name_scope('fully_connected2'):
        fc2 = nn_layer(fc1_dropout, 192, 'fully_connected2', tf.nn.relu)

      with tf.name_scope('dropout2'):
        fc2_dropout = tf.nn.dropout(fc2, keep_prob)

      with tf.name_scope('logits'):
        logits = nn_layer(fc2_dropout, 10, 'logits')

  return X, y, keep_prob, logits

def cnn_cifar10_nodropout_small(graph):
  with graph.as_default():
    with tf.name_scope('Input'):
      with tf.name_scope('X'):
        X = tf.placeholder(DTYPE, shape=[None, 32*32*3], name='X')
        X_reshaped = tf.reshape(X, shape=[-1, 32, 32, 3])

      with tf.name_scope('y'):
        y = tf.placeholder(tf.int32, shape=[None], name='y')

    with tf.name_scope('conv1'):
      conv1 = tf.layers.conv2d(X_reshaped,
                               filters=6,
                               kernel_size=5,
                               strides=1,
                               padding='valid',
                               activation=tf.nn.relu,
                               name='conv1'
                               )
      max_pool1 = tf.nn.max_pool(value=conv1,
                                 ksize=(1, 2, 2, 1),
                                 strides=(1, 2, 2, 1),
                                 padding='VALID',
                                 name='max_pool1')
    with tf.name_scope('conv2'):
      conv2 = tf.layers.conv2d(max_pool1,
                               filters=12,
                               kernel_size=3,
                               strides=1,
                               padding='valid',
                               activation=tf.nn.relu,
                               name='conv2')
      max_pool2 = tf.nn.max_pool(value=conv2,
                                 ksize=(1, 2, 2, 1),
                                 strides=(1, 2, 2, 1),
                                 padding='VALID',
                                 name='max_pool2')

    with tf.name_scope('logits'):
      flatten = tf.layers.Flatten()(max_pool2)
      logits = nn_layer(X=flatten, n_neurons=10, name='logits', activation=None)

  return X, y, logits

def cnn_cifar10_nodropout_small_batch_norm(graph):

  with graph.as_default():
    is_train = tf.placeholder(tf.bool, shape=(), name='is_train')

    with tf.name_scope('Input'):
      with tf.name_scope('X'):
        X = tf.placeholder(DTYPE, shape=[None, 32*32*3], name='X')
        X_reshaped = tf.reshape(X, shape=[-1, 32, 32, 3])

      with tf.name_scope('y'):
        y = tf.placeholder(tf.int32, shape=[None], name='y')
    
    with tf.name_scope('conv1'):
      conv1 = tf.layers.conv2d(X_reshaped,
                               filters=6,
                               kernel_size=5,
                               strides=1,
                               padding='valid',
                               activation=tf.nn.relu,
                               name='conv1'
                               )
      #print('conv1 type', conv1.dtype)
      #print('conv1', get_shape(conv1))
      max_pool1 = tf.nn.max_pool(value=conv1,
                                 ksize=(1, 2, 2, 1),
                                 strides=(1, 2, 2, 1),
                                 padding='VALID',
                                 name='max_pool1')
      #print('max_pool1', get_shape(max_pool1))
      #print('max_pool1 type', max_pool1.dtype)
      #batch_norm1 = tf.layers.batch_normalization(
      #    max_pool1, training=is_train)
      #print('batch_norm1 type', batch_norm1.dtype)
      #print('batch_norm1', get_shape(batch_norm1))
    with tf.name_scope('conv2'):
      conv2 = tf.layers.conv2d(max_pool1,
                               filters=12,
                               kernel_size=3,
                               strides=1,
                               padding='valid',
                               activation=tf.nn.relu,
                               name='conv2')
      #print('conv2', get_shape(conv2))
      #print('conv2 type', conv2.dtype)
      max_pool2 = tf.nn.max_pool(value=conv2,
                                 ksize=(1, 2, 2, 1),
                                 strides=(1, 2, 2, 1),
                                 padding='VALID',
                                 name='max_pool2')
      #print('max_pool2', get_shape(max_pool2))
      #print('max_poo2', max_pool2.dtype)
      #batch_norm2 = tf.layers.batch_normalization(
      #    max_pool2, training=is_train)
      #print('batch_norm2', get_shape(batch_norm2))
      #print('batch_norm2 type', batch_norm2.dtype)

      flatten = tf.layers.Flatten()(max_pool2)

    with tf.name_scope('logits'):
      logits = nn_layer(X=flatten, n_neurons=10, name='logits', activation=None, dtype=DTYPE)

  return X, y, is_train, logits,

def cnn_cifar10_nodropout_small(graph):
  with graph.as_default():
    with tf.name_scope('Input'):
      with tf.name_scope('X'):
        X = tf.placeholder(DTYPE, shape=[None, 32*32*3], name='X')
        X_reshaped = tf.reshape(X, shape=[-1, 32, 32, 3])

      with tf.name_scope('y'):
        y = tf.placeholder(tf.int32, shape=[None], name='y')

    with tf.name_scope('conv1'):
      conv1 = tf.layers.conv2d(X_reshaped,
                               filters=6,
                               kernel_size=5,
                               strides=1,
                               padding='valid',
                               activation=tf.nn.relu,
                               name='conv1'
                               )
      max_pool1 = tf.nn.max_pool(value=conv1,
                                 ksize=(1, 2, 2, 1),
                                 strides=(1, 2, 2, 1),
                                 padding='VALID',
                                 name='max_pool1')
    with tf.name_scope('conv2'):
      conv2 = tf.layers.conv2d(max_pool1,
                               filters=12,
                               kernel_size=3,
                               strides=1,
                               padding='valid',
                               activation=tf.nn.relu,
                               name='conv2')
      max_pool2 = tf.nn.max_pool(value=conv2,
                                 ksize=(1, 2, 2, 1),
                                 strides=(1, 2, 2, 1),
                                 padding='VALID',
                                 name='max_pool2')

    with tf.name_scope('logits'):
      flatten = tf.layers.Flatten()(max_pool2)
      logits = nn_layer(X=flatten, n_neurons=10, name='logits', activation=None)

  return X, y, logits

def cnn_cifar10_model4(graph):
  height = 32
  width = 32
  channels = 3
  n_inputs = height * width * channels

  conv1_fmaps = 32
  conv1_ksize = 3
  conv1_stride = 1
  conv1_pad = "SAME"

  conv2_fmaps = 64
  conv2_ksize = 3
  conv2_stride = 2
  conv2_pad = "SAME"

  pool3_fmaps = conv2_fmaps

  n_fc1 = 64
  n_outputs = 10

  gpu_device_name = _gpu_device_name(0)

  with graph.as_default():
    with tf.name_scope('Input'):
      with tf.name_scope('X'):
        X = tf.placeholder(DTYPE, shape=[None, n_inputs], 
          name='X')
        X_reshaped = tf.reshape(X, 
        shape=[-1, height, width, channels])
    with tf.name_scope('y'):
      y = tf.placeholder(tf.int32, shape=[None], name='y')
    with tf.device(gpu_device_name):
      with tf.name_scope('conv1'):
        conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, 
          kernel_size=conv1_ksize, strides=conv1_stride, 
          padding=conv1_pad, activation=tf.nn.relu, name='conv1')

      with tf.name_scope('conv2'):
        conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, 
          kernel_size=conv2_ksize, strides=conv2_stride, 
          padding=conv2_pad, activation=tf.nn.relu, name='conv2')

      with tf.name_scope('pool3'):
        pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], 
          strides=[1, 2, 2, 1], padding='VALID')
        s1, s2 = pool3.get_shape().as_list()[1], pool3.get_shape().as_list()[2]
        pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * s1 * s2])

      with tf.name_scope('fully_connected'):
        fc = nn_layer(pool3_flat, n_fc1, activation=tf.nn.relu, name='fc')

    with tf.device(gpu_device_name):
      with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(DTYPE, name='keep_prob')
        fc_dropout = tf.nn.dropout(fc, keep_prob)

      with tf.name_scope('logits'):
        logits = tf.layers.dense(fc_dropout, n_outputs, name='logits')

  return X, y, keep_prob, logits

def cnn_cifar10_model3(graph):
  height = 32
  width = 32
  channels = 3
  n_inputs = height * width * channels

  conv1_fmaps = 32
  conv1_ksize = 3
  conv1_stride = 1
  conv1_pad = "SAME"

  conv2_fmaps = 64
  conv2_ksize = 3
  conv2_stride = 2
  conv2_pad = "SAME"

  pool3_fmaps = conv2_fmaps

  n_fc1 = 64
  n_outputs = 10

  gpu_device_name = _gpu_device_name(0)

  with graph.as_default():
    with tf.name_scope('Input'):
      with tf.name_scope('X'):
        X = tf.placeholder(DTYPE, 
                  shape=[None, n_inputs], 
                  name='X')
        X_reshaped = tf.reshape(X, 
                    shape=[-1, height, width, channels])
      with tf.name_scope('y'):
        y = tf.placeholder( tf.int32, 
                  shape=[None], 
                  name='y')
    with tf.device(gpu_device_name):
      with tf.name_scope('conv1'):
        conv1 = tf.layers.conv2d( X_reshaped, 
                      filters=conv1_fmaps, 
                      kernel_size=conv1_ksize,
                      strides=conv1_stride, padding=conv1_pad,
                      activation=tf.nn.relu,
                      name='conv1')
      with tf.name_scope('conv2'):
        conv2 = tf.layers.conv2d( conv1,
                      filters=conv2_fmaps,
                      kernel_size=conv2_ksize,
                      strides=conv2_stride, 
                      padding=conv2_pad,
                      activation=tf.nn.relu,
                      name='conv2')

      with tf.name_scope('pool3'):
        pool3 = tf.nn.max_pool( conv2, 
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='VALID')
        pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 7 * 7])

    with tf.name_scope('fully_connected'):
      fc = nn_layer(pool3_flat, n_fc1, activation=tf.nn.relu, name='fc')

    with tf.device(gpu_device_name):
      with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(DTYPE, name='keep_prob')
        fc_dropout = tf.nn.dropout(fc, keep_prob)

      with tf.name_scope('logits'):
        logits = tf.layers.dense(fc_dropout, n_outputs, name='logits')

  return X, y, keep_prob, logits


def cnn_cifar10_model_no_dropout(graph):

  with graph.as_default():
    with tf.name_scope('Inputs'):
      with tf.name_scope('X'):
        X = tf.placeholder(DTYPE, shape=(None, 3072), name='X')
        X_reshaped = tf.reshape(X, shape=[-1, 32, 32, 3], name='X_reshaped')
        #X_reshaped = tf.reshape(X, shape=[-1, 32, 32, 3])
      with tf.name_scope('y'):
        y = tf.placeholder(tf.int64, shape=(None), name='y')

    with tf.name_scope('conv1'):
      kernel = _variable_with_weight_decay( name='kernel1',
                          shape=[5, 5, 3, 64],
                          stddev=5e-2,
                          wd=None)
      conv = tf.nn.conv2d(input=X_reshaped,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding='SAME')
      biases = tf.get_variable(name='biases1', shape=[64], initializer=tf.constant_initializer(0.0))
      pre_activation = tf.nn.bias_add(conv, biases)
      conv1 = tf.nn.relu(pre_activation)

    with tf.name_scope('pool1'):

      pool1 = tf.nn.max_pool( conv1, 
                  ksize=[1, 3, 3, 1],
                  strides=[1, 2, 2, 1],
                  padding='SAME',
                  name='pool1')

    with tf.name_scope('norm1'):
      norm1 = tf.nn.lrn(  pool1, 
                4, 
                bias=1.0, 
                alpha=0.001 / 9.0, 
                beta=0.75,
                name='norm1')

    with tf.name_scope('conv2'):
      kernel = _variable_with_weight_decay( 'kernel2',
                          shape=[5, 5, 64, 64],
                          stddev=5e-2,
                          wd=None)
      conv = tf.nn.conv2d(input=norm1,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding='SAME')
      biases = tf.get_variable(name='biases2', shape=[64], initializer=tf.constant_initializer(0.1))
      pre_activation = tf.nn.bias_add(conv, biases)
      conv2 = tf.nn.relu(pre_activation)

    with tf.name_scope('norm2'):
      norm2 = tf.nn.lrn(  conv2, 
                4, 
                bias=1.0, 
                alpha=0.001 / 9.0, 
                beta=0.75,
                name='norm2')

    with tf.name_scope('pool2'):
      pool2 = tf.nn.max_pool( norm2, 
                  ksize=[1, 3, 3, 1],
                  strides=[1, 2, 2, 1],
                  padding='SAME',
                  name='pool2')

    with tf.name_scope('fully_connected1'):
      shape = pool2.get_shape().as_list()[1:]
      len_ = int(np.prod(np.array(shape)))
      reshaped = tf.reshape(pool2, shape=[-1, len_])
      fc1 = nn_layer(reshaped, 384, 'fully_connected1', tf.nn.relu)

    with tf.name_scope('fully_connected2'):
      fc2 = nn_layer(fc1, 192, 'fully_connected2', tf.nn.relu)

    with tf.name_scope('logits'):
      logits = nn_layer(fc2, 10, 'logits')
  return X, y, logits

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
    decay is not added for this Variable.
  
  Returns:
    Variable Tensor
  """
  
  init_val = tf.truncated_normal_initializer(stddev=stddev, dtype=DTYPE)
  
  var = tf.get_variable(name, shape, initializer=init_val, dtype=DTYPE)
  '''
  var = tf.Variable(  initial_value=init_val,
            validate_shape=True, 
            dtype=DTYPE, 
            name=name)
  '''
  
  
  if wd is not None:

    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
  #tf.add_to_collection('losses', weight_decay)
  
  return var
