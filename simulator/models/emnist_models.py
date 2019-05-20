import sys
import os

import tensorflow as tf

from simulator.graph.device_placer import _gpu_device_name
from simulator.simulator_utils import DTYPE
from simulator.models.helpers import DEFAULT_INITIALIZER
from simulator.models import utils

def lenet5_with_dropout(graph, verbose=False):

  with graph.as_default():
    with tf.device('gpu:0'):
      is_train = tf.placeholder_with_default(False, shape=(), name='is_train')
      with tf.name_scope('inputs'):
        x = tf.placeholder(DTYPE, shape=(None, 28, 28, 1), name='x')
        y = tf.placeholder(tf.int32, shape=(None), name='y')
        keep_prob = tf.placeholder_with_default(input=1.0,
                                                shape=(),
                                                name='keep_prob')
        padded = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]])

      with tf.name_scope('C1-conv'):
        conv1 = tf.layers.conv2d(padded,
                                 filters=6,
                                 kernel_size=5,
                                 strides=(1, 1),
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        if verbose:
          print('C1-outshape:', conv1.get_shape().as_list())

      with tf.variable_scope('S2-avg_pool'):
        conv1 = utils.average_pooling_with_weights(value=conv1,
                                                   ksize=(1, 2, 2, 1),
                                                   strides=(1, 2, 2, 1),
                                                   padding='VALID',
                                                   activation=tf.nn.tanh)
        if verbose:
          print('S2-outshape:', conv1.get_shape().as_list())
      
      with tf.variable_scope('dropout1'):
        conv1 = tf.nn.dropout(conv1, keep_prob)

      with tf.variable_scope('C3-conv'):
        conv2 = tf.layers.conv2d(conv1,
                                 filters=16,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        if verbose:
          print('C3-outshape:', conv2.get_shape().as_list())

      with tf.variable_scope('S4-avg_pool'):
        conv2 = utils.average_pooling_with_weights(value=conv2,
                                                   ksize=(1, 2, 2, 1),
                                                   strides=(1, 2, 2, 1),
                                                   padding='VALID',
                                                   activation=tf.nn.tanh)
        if verbose:
          print('S4-outshape:', conv2.get_shape().as_list())

      with tf.variable_scope('dropout2'):
        conv2 = tf.nn.dropout(conv2, keep_prob)
      
      with tf.variable_scope('C5-conv'):
        conv3 = tf.layers.conv2d(conv2,
                                 filters=120,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)

        if verbose:
          print('C5-outshape:', conv3.get_shape().as_list())
        conv3 = tf.nn.dropout(conv3, keep_prob)

      with tf.name_scope('F6-fc'):
        flatten = tf.layers.Flatten()(conv3)
        fc1 = tf.layers.dense(inputs=flatten,
                              units=84,
                              activation=tf.nn.tanh,
                              kernel_initializer=DEFAULT_INITIALIZER,
                              name='fc2')
      with tf.name_scope('dropout3'):
        fc1 = tf.nn.dropout(fc1, keep_prob)
      
        if verbose:
          print('F6-outshape:', fc1.get_shape().as_list())

      with tf.name_scope('logits'):
        '''
        logits = tf.layers.dense(inputs=fc1,
                                 units=10,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        '''
        logits = utils.rbf_euclidean_layer(inputs=fc1, units=26)
        
    return x, y, is_train, keep_prob, logits

def lenet5_with_lr(graph, verbose=False):

  with graph.as_default():
    with tf.device('gpu:0'):
      is_train = tf.placeholder_with_default(False, shape=(), name='is_train')
      with tf.name_scope('inputs'):
        x = tf.placeholder(DTYPE, shape=(None, 28, 28, 1), name='x')
        y = tf.placeholder(tf.int32, shape=(None), name='y')
        lr = tf.placeholder_with_default(input=.001,
                                         shape=(),
                                         name='lr')
        padded = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]])

      with tf.name_scope('C1-conv'):
        conv1 = tf.layers.conv2d(padded,
                                 filters=6,
                                 kernel_size=5,
                                 strides=(1, 1),
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        if verbose:
          print('C1-outshape:', conv1.get_shape().as_list())

      with tf.variable_scope('S2-avg_pool'):
        conv1 = utils.average_pooling_with_weights(value=conv1,
                                                   ksize=(1, 2, 2, 1),
                                                   strides=(1, 2, 2, 1),
                                                   padding='VALID',
                                                   activation=tf.nn.tanh)
        if verbose:
          print('S2-outshape:', conv1.get_shape().as_list())
      
      with tf.variable_scope('C3-conv'):
        conv2 = tf.layers.conv2d(conv1,
                                 filters=16,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        if verbose:
          print('C3-outshape:', conv2.get_shape().as_list())

      with tf.variable_scope('S4-avg_pool'):
        conv2 = utils.average_pooling_with_weights(value=conv2,
                                                   ksize=(1, 2, 2, 1),
                                                   strides=(1, 2, 2, 1),
                                                   padding='VALID',
                                                   activation=tf.nn.tanh)
        if verbose:
          print('S4-outshape:', conv2.get_shape().as_list())

      
      with tf.variable_scope('C5-conv'):
        conv3 = tf.layers.conv2d(conv2,
                                 filters=120,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)

        if verbose:
          print('C5-outshape:', conv3.get_shape().as_list())


      with tf.name_scope('F6-fc'):
        flatten = tf.layers.Flatten()(conv3)
        fc1 = tf.layers.dense(inputs=flatten,
                              units=84,
                              activation=tf.nn.tanh,
                              kernel_initializer=DEFAULT_INITIALIZER,
                              name='fc2')

      
        if verbose:
          print('F6-outshape:', fc1.get_shape().as_list())

      with tf.name_scope('logits'):
        '''
        logits = tf.layers.dense(inputs=fc1,
                                 units=10,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        '''
        logits = utils.rbf_euclidean_layer(inputs=fc1, units=26)
        
  return x, y, is_train, lr, logits

def lenet5_with_lr_and_const_dropout(graph, verbose=False, keep_prob=0.9):

  with graph.as_default():
    with tf.device('gpu:0'):
      is_train = tf.placeholder_with_default(False, shape=(), name='is_train')
      with tf.name_scope('inputs'):
        x = tf.placeholder(DTYPE, shape=(None, 28, 28, 1), name='x')
        y = tf.placeholder(tf.int32, shape=(None), name='y')
        lr = tf.placeholder_with_default(input=.001,
                                         shape=(),
                                         name='lr')
        padded = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]])

      with tf.name_scope('C1-conv'):
        conv1 = tf.layers.conv2d(padded,
                                 filters=6,
                                 kernel_size=5,
                                 strides=(1, 1),
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        if verbose:
          print('C1-outshape:', conv1.get_shape().as_list())

      with tf.variable_scope('S2-avg_pool'):
        conv1 = utils.average_pooling_with_weights(value=conv1,
                                                   ksize=(1, 2, 2, 1),
                                                   strides=(1, 2, 2, 1),
                                                   padding='VALID',
                                                   activation=tf.nn.tanh)
        if verbose:
          print('S2-outshape:', conv1.get_shape().as_list())
      with tf.variable_scope('dropout1'):
        conv1 = tf.nn.dropout(conv1, keep_prob)

      with tf.variable_scope('C3-conv'):
        conv2 = tf.layers.conv2d(conv1,
                                 filters=16,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        if verbose:
          print('C3-outshape:', conv2.get_shape().as_list())

      with tf.variable_scope('S4-avg_pool'):
        conv2 = utils.average_pooling_with_weights(value=conv2,
                                                   ksize=(1, 2, 2, 1),
                                                   strides=(1, 2, 2, 1),
                                                   padding='VALID',
                                                   activation=tf.nn.tanh)
        if verbose:
          print('S4-outshape:', conv2.get_shape().as_list())

      with tf.variable_scope('dropout2'):
        conv2 = tf.nn.dropout(conv1, keep_prob)

      with tf.variable_scope('C5-conv'):
        conv3 = tf.layers.conv2d(conv2,
                                 filters=120,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)

        if verbose:
          print('C5-outshape:', conv3.get_shape().as_list())


      with tf.name_scope('F6-fc'):
        flatten = tf.layers.Flatten()(conv3)
        fc1 = tf.layers.dense(inputs=flatten,
                              units=84,
                              activation=tf.nn.tanh,
                              kernel_initializer=DEFAULT_INITIALIZER,
                              name='fc2')

        
        if verbose:
          print('F6-outshape:', fc1.get_shape().as_list())

      with tf.variable_scope('dropout'):
        fc1 = tf.nn.dropout(fc1, keep_prob)

      with tf.name_scope('logits'):
        '''
        logits = tf.layers.dense(inputs=fc1,
                                 units=10,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        '''
        logits = utils.rbf_euclidean_layer(inputs=fc1, units=26)
        
  return x, y, is_train, lr, logits

def lenet5_with_input_noise(graph, verbose=False):

  with graph.as_default():
    with tf.device('gpu:0'):
      is_train = tf.placeholder_with_default(False, shape=(), name='is_train')
      with tf.name_scope('inputs'):
        x = tf.placeholder(DTYPE, shape=(None, 28, 28, 1), name='x')
        y = tf.placeholder(tf.int32, shape=(None), name='y')
        stddev_ph = tf.placeholder_with_default(input=.001,
                                                shape=(),
                                                name='stddev')
        
        noise_inputs = utils.apply_gaussian_noise(x, is_train, stddev_ph)
        padded = tf.pad(noise_inputs, [[0, 0], [2, 2], [2, 2], [0, 0]])

      with tf.name_scope('C1-conv'):
        conv1 = tf.layers.conv2d(padded,
                                 filters=6,
                                 kernel_size=5,
                                 strides=(1, 1),
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        if verbose:
          print('C1-outshape:', conv1.get_shape().as_list())

      with tf.variable_scope('S2-avg_pool'):
        conv1 = utils.average_pooling_with_weights(value=conv1,
                                                   ksize=(1, 2, 2, 1),
                                                   strides=(1, 2, 2, 1),
                                                   padding='VALID',
                                                   activation=tf.nn.tanh)
        if verbose:
          print('S2-outshape:', conv1.get_shape().as_list())
      
      with tf.variable_scope('C3-conv'):
        conv2 = tf.layers.conv2d(conv1,
                                 filters=16,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        if verbose:
          print('C3-outshape:', conv2.get_shape().as_list())

      with tf.variable_scope('S4-avg_pool'):
        conv2 = utils.average_pooling_with_weights(value=conv2,
                                                   ksize=(1, 2, 2, 1),
                                                   strides=(1, 2, 2, 1),
                                                   padding='VALID',
                                                   activation=tf.nn.tanh)
        if verbose:
          print('S4-outshape:', conv2.get_shape().as_list())

      
      with tf.variable_scope('C5-conv'):
        conv3 = tf.layers.conv2d(conv2,
                                 filters=120,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)

        if verbose:
          print('C5-outshape:', conv3.get_shape().as_list())


      with tf.name_scope('F6-fc'):
        flatten = tf.layers.Flatten()(conv3)
        fc1 = tf.layers.dense(inputs=flatten,
                              units=84,
                              activation=tf.nn.tanh,
                              kernel_initializer=DEFAULT_INITIALIZER,
                              name='fc2')

      
        if verbose:
          print('F6-outshape:', fc1.get_shape().as_list())

      with tf.name_scope('logits'):
        '''
        logits = tf.layers.dense(inputs=fc1,
                                 units=10,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        '''
        logits = utils.rbf_euclidean_layer(inputs=fc1, units=26)
        
  return x, y, is_train, stddev_ph, logits

def lenet5_with_input_noise_const_dropout(graph, keep_prob=0.9, verbose=False):

  with graph.as_default():
    with tf.device('gpu:0'):
      is_train = tf.placeholder_with_default(False, shape=(), name='is_train')
      with tf.name_scope('inputs'):
        x = tf.placeholder(DTYPE, shape=(None, 28, 28, 1), name='x')
        y = tf.placeholder(tf.int32, shape=(None), name='y')
        stddev_ph = tf.placeholder_with_default(input=.001,
                                                shape=(),
                                                name='stddev')
        
        noise_inputs = utils.apply_gaussian_noise(x, is_train, stddev_ph)
        padded = tf.pad(noise_inputs, [[0, 0], [2, 2], [2, 2], [0, 0]])

      with tf.name_scope('C1-conv'):
        conv1 = tf.layers.conv2d(padded,
                                 filters=6,
                                 kernel_size=5,
                                 strides=(1, 1),
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        if verbose:
          print('C1-outshape:', conv1.get_shape().as_list())

      with tf.variable_scope('S2-avg_pool'):
        conv1 = utils.average_pooling_with_weights(value=conv1,
                                                   ksize=(1, 2, 2, 1),
                                                   strides=(1, 2, 2, 1),
                                                   padding='VALID',
                                                   activation=tf.nn.tanh)
        conv1 = tf.nn.dropout(conv1, keep_prob)
        if verbose:
          print('S2-outshape:', conv1.get_shape().as_list())
      
      with tf.variable_scope('C3-conv'):
        conv2 = tf.layers.conv2d(conv1,
                                 filters=16,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        if verbose:
          print('C3-outshape:', conv2.get_shape().as_list())

      with tf.variable_scope('S4-avg_pool'):
        conv2 = utils.average_pooling_with_weights(value=conv2,
                                                   ksize=(1, 2, 2, 1),
                                                   strides=(1, 2, 2, 1),
                                                   padding='VALID',
                                                   activation=tf.nn.tanh)
        conv2 = tf.nn.dropout(conv2, keep_prob)
        if verbose:
          print('S4-outshape:', conv2.get_shape().as_list())

      
      with tf.variable_scope('C5-conv'):
        conv3 = tf.layers.conv2d(conv2,
                                 filters=120,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)

        if verbose:
          print('C5-outshape:', conv3.get_shape().as_list())


      with tf.name_scope('F6-fc'):
        flatten = tf.layers.Flatten()(conv3)
        fc1 = tf.layers.dense(inputs=flatten,
                              units=84,
                              activation=tf.nn.tanh,
                              kernel_initializer=DEFAULT_INITIALIZER,
                              name='fc2')
        fc1 = tf.nn.dropout(fc1, keep_prob)
      
        if verbose:
          print('F6-outshape:', fc1.get_shape().as_list())

      with tf.name_scope('logits'):
        '''
        logits = tf.layers.dense(inputs=fc1,
                                 units=10,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        '''
        logits = utils.rbf_euclidean_layer(inputs=fc1, units=26)
        
  return x, y, is_train, stddev_ph, logits

def lenet5_with_weight_noise(graph, verbose=False):

  with graph.as_default():
    with tf.device('gpu:0'):
      is_train = tf.placeholder_with_default(False, shape=(), name='is_train')
      with tf.name_scope('inputs'):
        x = tf.placeholder(DTYPE, shape=(None, 28, 28, 1), name='x')
        y = tf.placeholder(tf.int32, shape=(None), name='y')
        stddev_ph = tf.placeholder_with_default(input=.001,
                                                shape=(),
                                                name='stddev')

        padded = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]])

      with tf.name_scope('C1-conv'):
        conv1 = tf.layers.conv2d(padded,
                                 filters=6,
                                 kernel_size=5,
                                 strides=(1, 1),
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        if verbose:
          print('C1-outshape:', conv1.get_shape().as_list())

      with tf.variable_scope('S2-avg_pool'):
        conv1 = utils.average_pooling_with_weights(value=conv1,
                                                   ksize=(1, 2, 2, 1),
                                                   strides=(1, 2, 2, 1),
                                                   padding='VALID',
                                                   activation=tf.nn.tanh)
        if verbose:
          print('S2-outshape:', conv1.get_shape().as_list())
      
      with tf.variable_scope('C3-conv'):
        conv2 = tf.layers.conv2d(conv1,
                                 filters=16,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        if verbose:
          print('C3-outshape:', conv2.get_shape().as_list())

      with tf.variable_scope('S4-avg_pool'):
        conv2 = utils.average_pooling_with_weights(value=conv2,
                                                   ksize=(1, 2, 2, 1),
                                                   strides=(1, 2, 2, 1),
                                                   padding='VALID',
                                                   activation=tf.nn.tanh)
        if verbose:
          print('S4-outshape:', conv2.get_shape().as_list())

      
      with tf.variable_scope('C5-conv'):
        conv3 = tf.layers.conv2d(conv2,
                                 filters=120,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)

        if verbose:
          print('C5-outshape:', conv3.get_shape().as_list())

      with tf.name_scope('F6-fc'):
        flatten = tf.layers.Flatten()(conv3)
        fc1 = tf.layers.dense(inputs=flatten,
                              units=84,
                              activation=tf.nn.tanh,
                              kernel_initializer=DEFAULT_INITIALIZER,
                              name='fc2')

      
        if verbose:
          print('F6-outshape:', fc1.get_shape().as_list())

      with tf.name_scope('logits'):
        '''
        logits = tf.layers.dense(inputs=fc1,
                                 units=10,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        '''
        logits = utils.rbf_euclidean_layer(inputs=fc1, units=26)
        
  return x, y, is_train, stddev_ph, logits

def lenet5(graph, verbose=False):
  # for regularization
  with graph.as_default():
    with tf.device('gpu:0'):
      is_train = tf.placeholder_with_default(False, shape=(), name='is_train')
      with tf.name_scope('inputs'):
        x = tf.placeholder(DTYPE, shape=(None, 28, 28, 1), name='x')
        y = tf.placeholder(tf.int32, shape=(None), name='y')


        padded = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]])

      with tf.name_scope('C1-conv'):
        conv1 = tf.layers.conv2d(padded,
                                 filters=6,
                                 kernel_size=5,
                                 strides=(1, 1),
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        if verbose:
          print('C1-outshape:', conv1.get_shape().as_list())

      with tf.variable_scope('S2-avg_pool'):
        conv1 = utils.average_pooling_with_weights(value=conv1,
                                                   ksize=(1, 2, 2, 1),
                                                   strides=(1, 2, 2, 1),
                                                   padding='VALID',
                                                   activation=tf.nn.tanh)
        if verbose:
          print('S2-outshape:', conv1.get_shape().as_list())
      
      with tf.variable_scope('C3-conv'):
        conv2 = tf.layers.conv2d(conv1,
                                 filters=16,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        if verbose:
          print('C3-outshape:', conv2.get_shape().as_list())

      with tf.variable_scope('S4-avg_pool'):
        conv2 = utils.average_pooling_with_weights(value=conv2,
                                                   ksize=(1, 2, 2, 1),
                                                   strides=(1, 2, 2, 1),
                                                   padding='VALID',
                                                   activation=tf.nn.tanh)
        if verbose:
          print('S4-outshape:', conv2.get_shape().as_list())

      
      with tf.variable_scope('C5-conv'):
        conv3 = tf.layers.conv2d(conv2,
                                 filters=120,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=DEFAULT_INITIALIZER)

        if verbose:
          print('C5-outshape:', conv3.get_shape().as_list())

      with tf.name_scope('F6-fc'):
        flatten = tf.layers.Flatten()(conv3)
        fc1 = tf.layers.dense(inputs=flatten,
                              units=84,
                              activation=tf.nn.tanh,
                              kernel_initializer=DEFAULT_INITIALIZER,
                              name='fc2')

      
        if verbose:
          print('F6-outshape:', fc1.get_shape().as_list())

      with tf.name_scope('logits'):
        '''
        logits = tf.layers.dense(inputs=fc1,
                                 units=10,
                                 kernel_initializer=DEFAULT_INITIALIZER)
        '''
        logits = utils.rbf_euclidean_layer(inputs=fc1, units=26)
        
  return x, y, is_train, logits



