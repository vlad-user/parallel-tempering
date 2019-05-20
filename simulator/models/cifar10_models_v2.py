import sys
import os

import tensorflow as tf

from simulator.graph.device_placer import _gpu_device_name
from simulator.simulator_utils import DTYPE
from simulator.models.helpers import DEFAULT_INITIALIZER
from simulator.models import utils

def lenet5_with_dropout_original(graph, verbose=False):

  with graph.as_default():
    with tf.device('gpu:0'):
      is_train = tf.placeholder_with_default(False, shape=(), name='is_train')
      with tf.name_scope('inputs'):
        x = tf.placeholder(DTYPE, shape=(None, 32, 32, 3), name='x')
        y = tf.placeholder(tf.int32, shape=(None), name='y')
        keep_prob = tf.placeholder_with_default(input=1.0,
                                                shape=(),
                                                name='keep_prob')
      with tf.name_scope('augmentation'):
        augmented = utils.augment_images(x, is_train)

      with tf.name_scope('C1-conv'):
        conv1 = tf.layers.conv2d(augmented,
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
        logits = utils.rbf_euclidean_layer(inputs=fc1, units=10)
        
    return x, y, is_train, keep_prob, logits

def lenet5_with_dropout(graph):
  with graph.as_default():
    istrain = tf.placeholder(tf.bool, shape=(), name='is_train')
    with tf.device('gpu:0'):
      with tf.name_scope('inputs'):
        x = tf.placeholder(DTYPE, shape=(None, 32, 32, 3))
        y = tf.placeholder(tf.int32, shape=(None))
        keep_prob = tf.placeholder_with_default(input=1.0,
                                                shape=(),
                                                name='keep_prob')

      with tf.name_scope('conv1'):
        conv1 = tf.layers.conv2d(x,
                                 filters=6,
                                 kernel_size=5,
                                 strides=(1, 1),
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=DEFAULT_INITIALIZER)

      with tf.name_scope('max_pool1'):
        conv1 = tf.nn.max_pool(value=conv1,
                               ksize=(1, 2, 2, 1),
                               strides=(1, 2, 2, 1),
                               padding='VALID')

      with tf.name_scope('dropout1'):
        conv1 = tf.nn.dropout(conv1, keep_prob)

      with tf.name_scope('conv2'):
        conv2 = tf.layers.conv2d(x,
                                 filters=16,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=DEFAULT_INITIALIZER)

      with tf.name_scope('max_pool2'):
        conv2 = tf.nn.max_pool(value=conv2,
                               ksize=(1, 2, 2, 1),
                               strides=(1, 2, 2, 1),
                               padding='VALID',
                               name='max_pool2')

      with tf.name_scope('dropout2'):
        conv2 = tf.nn.dropout(conv2, keep_prob)

      with tf.name_scope('conv3'):
        conv3 = tf.layers.conv2d(conv2,
                                 filters=120,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=DEFAULT_INITIALIZER)
      with tf.name_scope('fc1'):
        flattened = tf.layers.Flatten()(conv3)
        fc1 = tf.layers.dense(flattened, 84, activation=tf.nn.relu, kernel_initializer=DEFAULT_INITIALIZER)
        fc1 = tf.nn.dropout(flattened, keep_prob)

      with tf.name_scope('logits'):
        logits = tf.layers.dense(inputs=fc1,
                                 units=10,
                                 activation=None,
                                 kernel_initializer=DEFAULT_INITIALIZER)

  return x, y, istrain, keep_prob, logits

def lenet5_with_dropout(graph):
  with graph.as_default():
    istrain = tf.placeholder(tf.bool, shape=(), name='is_train')
    with tf.device('gpu:0'):
      with tf.name_scope('inputs'):
        x = tf.placeholder(DTYPE, shape=(None, 32, 32, 3))
        y = tf.placeholder(tf.int32, shape=(None))
        keep_prob = tf.placeholder_with_default(input=1.0,
                                                shape=(),
                                                name='keep_prob')

      with tf.name_scope('conv1'):
        conv1 = tf.layers.conv2d(x,
                                 filters=6,
                                 kernel_size=5,
                                 strides=(1, 1),
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=DEFAULT_INITIALIZER)

      with tf.name_scope('max_pool1'):
        conv1 = tf.nn.max_pool(value=conv1,
                               ksize=(1, 2, 2, 1),
                               strides=(1, 2, 2, 1),
                               padding='VALID')

      with tf.name_scope('dropout1'):
        conv1 = tf.nn.dropout(conv1, keep_prob)

      with tf.name_scope('conv2'):
        conv2 = tf.layers.conv2d(x,
                                 filters=16,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=DEFAULT_INITIALIZER)

      with tf.name_scope('max_pool2'):
        conv2 = tf.nn.max_pool(value=conv2,
                               ksize=(1, 2, 2, 1),
                               strides=(1, 2, 2, 1),
                               padding='VALID',
                               name='max_pool2')

      with tf.name_scope('dropout2'):
        conv2 = tf.nn.dropout(conv2, keep_prob)

      with tf.name_scope('conv3'):
        conv3 = tf.layers.conv2d(conv2,
                                 filters=120,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=DEFAULT_INITIALIZER)
      with tf.name_scope('fc1'):
        flattened = tf.layers.Flatten()(conv3)
        fc1 = tf.layers.dense(flattened, 84, activation=tf.nn.relu, kernel_initializer=DEFAULT_INITIALIZER)
        fc1 = tf.nn.dropout(flattened, keep_prob)

      with tf.name_scope('logits'):
        logits = tf.layers.dense(inputs=fc1,
                                 units=10,
                                 activation=None,
                                 kernel_initializer=DEFAULT_INITIALIZER)

  return x, y, istrain, keep_prob, logits

def lenet5_lr_with_const_dropout(graph, keep_prob=0.6):
  with graph.as_default():
    istrain = tf.placeholder(tf.bool, shape=(), name='is_train')
    with tf.device('gpu:0'):
      with tf.name_scope('inputs'):
        x = tf.placeholder(DTYPE, shape=(None, 32, 32, 3))
        y = tf.placeholder(tf.int32, shape=(None))
        lr = tf.placeholder(tf.float32, shape=(), name='lr')

      with tf.name_scope('conv1'):
        conv1 = tf.layers.conv2d(x,
                                 filters=6,
                                 kernel_size=5,
                                 strides=(1, 1),
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=DEFAULT_INITIALIZER)

      with tf.name_scope('max_pool1'):
        conv1 = tf.nn.max_pool(value=conv1,
                               ksize=(1, 2, 2, 1),
                               strides=(1, 2, 2, 1),
                               padding='VALID')

      with tf.name_scope('dropout1'):
        conv1 = tf.nn.dropout(conv1, keep_prob)

      with tf.name_scope('conv2'):
        conv2 = tf.layers.conv2d(x,
                                 filters=16,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=DEFAULT_INITIALIZER)

      with tf.name_scope('max_pool2'):
        conv2 = tf.nn.max_pool(value=conv2,
                               ksize=(1, 2, 2, 1),
                               strides=(1, 2, 2, 1),
                               padding='VALID',
                               name='max_pool2')

      with tf.name_scope('dropout2'):
        conv2 = tf.nn.dropout(conv2, keep_prob)

      with tf.name_scope('conv3'):
        conv3 = tf.layers.conv2d(conv2,
                                 filters=120,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=DEFAULT_INITIALIZER)
      with tf.name_scope('fc1'):
        flattened = tf.layers.Flatten()(conv3)
        fc1 = tf.layers.dense(flattened, 84, activation=tf.nn.relu, kernel_initializer=DEFAULT_INITIALIZER)
        fc1 = tf.nn.dropout(flattened, keep_prob)

      with tf.name_scope('logits'):
        logits = tf.layers.dense(inputs=fc1,
                                 units=10,
                                 activation=None,
                                 kernel_initializer=DEFAULT_INITIALIZER)

  return x, y, istrain, lr, logits

def lenet5(graph,):
  x, y, istrain, _, logits = lenet5_lr_with_const_dropout(graph, keep_prob=1.0)
  return x, y, istrain, logits

def lenet5_l2_with_const_dropout(graph):
  x, y, istrain, _, logits = lenet5_lr_with_const_dropout(graph, keep_prob=0.7)
  return x, y, istrain, logits

def lenet5_with_input_noise(graph, verbose=False):

  with graph.as_default():
    with tf.device('gpu:0'):
      is_train = tf.placeholder_with_default(False, shape=(), name='is_train')
      with tf.name_scope('inputs'):
        x = tf.placeholder(DTYPE, shape=(None, 32, 32, 3), name='x')
        y = tf.placeholder(tf.int32, shape=(None), name='y')
        stddev_ph = tf.placeholder_with_default(input=.001,
                                                shape=(),
                                                name='stddev')
        
        noisy_inputs = utils.apply_gaussian_noise(x, is_train, stddev_ph)

      with tf.name_scope('C1-conv'):
        conv1 = tf.layers.conv2d(noisy_inputs,
                                 filters=6,
                                 kernel_size=5,
                                 strides=(1, 1),
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=DEFAULT_INITIALIZER)

      with tf.name_scope('max_pool1'):
        conv1 = tf.nn.max_pool(value=conv1,
                               ksize=(1, 2, 2, 1),
                               strides=(1, 2, 2, 1),
                               padding='VALID')
      
      with tf.variable_scope('C3-conv'):
        conv2 = tf.layers.conv2d(conv1,
                                 filters=16,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=DEFAULT_INITIALIZER)


      with tf.name_scope('max_pool2'):
        conv2 = tf.nn.max_pool(value=conv2,
                               ksize=(1, 2, 2, 1),
                               strides=(1, 2, 2, 1),
                               padding='VALID',
                               name='max_pool2')

      
      with tf.variable_scope('C5-conv'):
        conv3 = tf.layers.conv2d(conv2,
                                 filters=120,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=DEFAULT_INITIALIZER)

      with tf.name_scope('F6-fc'):
        flatten = tf.layers.Flatten()(conv3)
        fc1 = tf.layers.dense(inputs=flatten,
                              units=84,
                              activation=tf.nn.relu,
                              kernel_initializer=DEFAULT_INITIALIZER,
                              name='fc2')
        fc1 = tf.nn.dropout(fc1, keep_prob)

      with tf.name_scope('logits'):

        logits = tf.layers.dense(inputs=fc1,
                                 units=10,
                                 kernel_initializer=DEFAULT_INITIALIZER)

  return x, y, is_train, stddev_ph, logits

def lenet5_with_input_noise_const_dropout(graph, keep_prob=0.7, verbose=False):

  with graph.as_default():
    with tf.device('gpu:0'):
      is_train = tf.placeholder_with_default(False, shape=(), name='is_train')
      with tf.name_scope('inputs'):
        x = tf.placeholder(DTYPE, shape=(None, 32, 32, 3), name='x')
        y = tf.placeholder(tf.int32, shape=(None), name='y')
        stddev_ph = tf.placeholder_with_default(input=.001,
                                                shape=(),
                                                name='stddev')
        
        noisy_inputs = utils.apply_gaussian_noise(x, is_train, stddev_ph)

      with tf.name_scope('C1-conv'):
        conv1 = tf.layers.conv2d(noisy_inputs,
                                 filters=6,
                                 kernel_size=5,
                                 strides=(1, 1),
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=DEFAULT_INITIALIZER)

      with tf.name_scope('max_pool1'):
        conv1 = tf.nn.max_pool(value=conv1,
                               ksize=(1, 2, 2, 1),
                               strides=(1, 2, 2, 1),
                               padding='VALID')
        conv1 = tf.nn.dropout(conv1, keep_prob)
      with tf.variable_scope('C3-conv'):
        conv2 = tf.layers.conv2d(conv1,
                                 filters=16,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=DEFAULT_INITIALIZER)


      with tf.name_scope('max_pool2'):
        conv2 = tf.nn.max_pool(value=conv2,
                               ksize=(1, 2, 2, 1),
                               strides=(1, 2, 2, 1),
                               padding='VALID',
                               name='max_pool2')
        conv2 = tf.nn.dropout(conv2, keep_prob)
      
      with tf.variable_scope('C5-conv'):
        conv3 = tf.layers.conv2d(conv2,
                                 filters=120,
                                 kernel_size=5,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=DEFAULT_INITIALIZER)

      with tf.name_scope('F6-fc'):
        flatten = tf.layers.Flatten()(conv3)
        fc1 = tf.layers.dense(inputs=flatten,
                              units=84,
                              activation=tf.nn.relu,
                              kernel_initializer=DEFAULT_INITIALIZER,
                              name='fc2')
        fc1 = tf.nn.dropout(fc1, keep_prob)
      with tf.name_scope('logits'):

        logits = tf.layers.dense(inputs=fc1,
                                 units=10,
                                 kernel_initializer=DEFAULT_INITIALIZER)

  return x, y, is_train, stddev_ph, logits

if __name__ == '__main__':
  os.system('clear')
  lenet5_with_dropout(tf.Graph(), verbose=True)
