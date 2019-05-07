"""NN models for mnist"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib # pylint: disable=no-name-in-module

from simulator.graph.device_placer import _gpu_device_name
from simulator.simulator_utils import DTYPE
from simulator.models.helpers import nn_layer, flatten

from simulator.models.helpers import DEFAULT_INITIALIZER

def lenet5_lr(graph):
  with graph.as_default():
    is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
    with tf.name_scope('inputs'):
      x = tf.placeholder(DTYPE, shape=[None, 28*28*1], name='x')
      x_reshaped = tf.reshape(x, shape=[-1, 28, 28, 1])
      y = tf.placeholder(tf.int32, shape=[None], name='y')
      lr = tf.placeholder(tf.float32, shape=(), name='learning_rate')
    with tf.name_scope('conv1'):
      conv1 = tf.layers.conv2d(x_reshaped,
                               filters=6,
                               kernel_size=5,
                               strides=(1, 1),
                               padding='VALID',
                               activation=tf.nn.relu,
                               kernel_initializer=DEFAULT_INITIALIZER)
      conv1 = tf.nn.max_pool(value=conv1,
                             ksize=(1, 2, 2, 1),
                             strides=(1, 2, 2, 1),
                             padding='VALID',
                             name='max_pool1')
    with tf.name_scope('conv2'):
      conv2 = tf.layers.conv2d(conv1,
                               filters=16,
                               kernel_size=5,
                               padding='VALID',
                               activation=tf.nn.relu,
                               kernel_initializer=DEFAULT_INITIALIZER)
      conv2 = tf.nn.max_pool(value=conv2,
                             ksize=(1, 2, 2, 1),
                             strides=(1, 2, 2, 1),
                             padding='VALID',
                             name='max_pool2')
    with tf.name_scope('fc1'):
      flatten = tf.layers.Flatten()(conv2)
      fc1 = tf.layers.dense(inputs=flatten,
                            units=120,
                            activation=tf.nn.relu,
                            kernel_initializer=DEFAULT_INITIALIZER,
                            name='fc1')

    with tf.name_scope('fc2'):
      fc2 = tf.layers.dense(inputs=fc1,
                            units=84,
                            activation=tf.nn.relu,
                            kernel_initializer=DEFAULT_INITIALIZER,
                            name='fc2')

    with tf.name_scope('logits'):
      logits = tf.layers.dense(inputs=fc2,
                               units=10,
                               activation=None,
                               kernel_initializer=DEFAULT_INITIALIZER,
                               name='logits')

    return x, y, is_train, lr, logits

def lenet5_with_dropout(graph):
  with graph.as_default():
    is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
    with tf.name_scope('inputs'):
      x = tf.placeholder(DTYPE, shape=[None, 28*28*1], name='x')
      x_reshaped = tf.reshape(x, shape=[-1, 28, 28, 1])
      y = tf.placeholder(tf.int32, shape=[None], name='y')
      keep_prob = tf.placeholder_with_default(input=1.0,
                                              shape=(),
                                              name='keep_prob')

    with tf.name_scope('conv1'):
      conv1 = tf.layers.conv2d(x_reshaped,
                               filters=6,
                               kernel_size=5,
                               strides=(1, 1),
                               padding='VALID',
                               activation=tf.nn.relu,
                               kernel_initializer=DEFAULT_INITIALIZER)
      conv1 = tf.nn.max_pool(value=conv1,
                             ksize=(1, 2, 2, 1),
                             strides=(1, 2, 2, 1),
                             padding='VALID',
                             name='max_pool1')
      conv1 = tf.nn.dropout(conv1, keep_prob)

    with tf.name_scope('conv2'):
      conv2 = tf.layers.conv2d(conv1,
                               filters=16,
                               kernel_size=5,
                               padding='VALID',
                               activation=tf.nn.relu,
                               kernel_initializer=DEFAULT_INITIALIZER)
      conv2 = tf.nn.max_pool(value=conv2,
                             ksize=(1, 2, 2, 1),
                             strides=(1, 2, 2, 1),
                             padding='VALID',
                             name='max_pool2')
      conv2 = tf.nn.dropout(conv2, keep_prob)
    with tf.name_scope('fc1'):
      flatten = tf.layers.Flatten()(conv2)
      fc1 = tf.layers.dense(inputs=flatten,
                            units=120,
                            activation=tf.nn.relu,
                            kernel_initializer=DEFAULT_INITIALIZER,
                            name='fc1')
      fc1 = tf.nn.dropout(fc1, keep_prob)

    with tf.name_scope('fc2'):
      fc2 = tf.layers.dense(inputs=fc1,
                            units=84,
                            activation=tf.nn.relu,
                            kernel_initializer=DEFAULT_INITIALIZER,
                            name='fc2')
      fc2 = tf.nn.dropout(fc2, keep_prob)

    with tf.name_scope('logits'):
      logits = tf.layers.dense(inputs=fc2,
                               units=10,
                               activation=None,
                               kernel_initializer=DEFAULT_INITIALIZER,
                               name='logits')

    return x, y, is_train, keep_prob, logits

def lenet1(graph):
  """Lenet-5, no dropout"""
  with graph.as_default():
    is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
    with tf.name_scope('Input'):
      with tf.name_scope('X'):

        X = tf.placeholder(DTYPE, shape=[None, 28*28], name='X')
        X_reshaped = tf.reshape(X, shape=[-1, 28, 28, 1])

      with tf.name_scope('y'):
        y = tf.placeholder(tf.int32, shape=[None], name='y')
    with tf.device(_gpu_device_name(0)):
      with tf.name_scope('conv1'):
        conv1 = tf.layers.conv2d(X_reshaped,
                                 filters=6,
                                 kernel_size=5,
                                 strides=1,
                                 padding='valid',
                                 kernel_initializer=DEFAULT_INITIALIZER,
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
                                 kernel_initializer=DEFAULT_INITIALIZER,
                                 name='conv2')
        max_pool2 = tf.nn.max_pool(value=conv2,
                                   ksize=(1, 2, 2, 1),
                                   strides=(1, 2, 2, 1),
                                   padding='VALID',
                                   name='max_pool2')

      with tf.name_scope('logits'):
        flatten = tf.layers.Flatten()(max_pool2)
        logits = tf.layers.dense(flatten,
                                 units=10,
                                 kernel_initializer=DEFAULT_INITIALIZER,
                                 name='logits')


  return X, y, is_train, logits

def lenet1_dropout(graph):
  """Lenet-5 with dropout."""
  with graph.as_default():
    is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
    #keep_prob = tf.placeholder(DTYPE, shape=(), name='keep_prob')
    keep_prob = tf.placeholder_with_default(1.0, shape=())
    with tf.name_scope('Input'):
      with tf.name_scope('X'):

        X = tf.placeholder(DTYPE, shape=[None, 28*28], name='X')
        X_reshaped = tf.reshape(X, shape=[-1, 28, 28, 1])

      with tf.name_scope('y'):
        y = tf.placeholder(tf.int32, shape=[None], name='y')
    with tf.device(_gpu_device_name(0)):
      with tf.name_scope('conv1'):
        conv1 = tf.layers.conv2d(X_reshaped,
                                 filters=6,
                                 kernel_size=5,
                                 strides=1,
                                 padding='valid',
                                 activation=tf.nn.relu,
                                 kernel_initializer=DEFAULT_INITIALIZER,
                                 name='conv1'
                                 )
        max_pool1 = tf.nn.max_pool(value=conv1,
                                   ksize=(1, 2, 2, 1),
                                   strides=(1, 2, 2, 1),
                                   padding='VALID',
                                   name='max_pool1')
        max_pool1 = tf.nn.dropout(max_pool1, keep_prob)
      with tf.name_scope('conv2'):
        conv2 = tf.layers.conv2d(max_pool1,
                                 filters=12,
                                 kernel_size=3,
                                 strides=1,
                                 padding='valid',
                                 activation=tf.nn.relu,
                                 kernel_initializer=DEFAULT_INITIALIZER,
                                 name='conv2')
        max_pool2 = tf.nn.max_pool(value=conv2,
                                   ksize=(1, 2, 2, 1),
                                   strides=(1, 2, 2, 1),
                                   padding='VALID',
                                   name='max_pool2')
        max_pool2 = tf.nn.dropout(max_pool2, keep_prob)

      with tf.name_scope('logits'):
        flatten = tf.layers.Flatten()(max_pool2)
        logits = tf.layers.dense(flatten,
                                 units=10,
                                 kernel_initializer=DEFAULT_INITIALIZER,
                                 name='logits')

  return X, y, is_train, keep_prob, logits


############################ WORKING MODELS ############################

def nn_mnist_model_small(graph):
  with graph.as_default():
    is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
  n_inputs = 28*28
  n_hidden1 = 5
  n_hidden2 = 5
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

    logits = nn_layer(
        hidden2,
        n_outputs,
        name='logits')

  return X, y, is_train, logits

def nn_mnist_model_small_dropout(graph):
  with graph.as_default():
    is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
    keep_prob = tf.placeholder(DTYPE,
                               shape=(),
                               name='keep_prob')

  n_inputs = 28*28
  n_hidden1 = 5
  n_hidden2 = 5
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
        activation=tf.nn.relu)

    hidden1_dropout = tf.nn.dropout(hidden1, keep_prob)

    hidden2 = nn_layer(
        hidden1_dropout,
        n_hidden2,
        name='hidden2',
        activation=tf.nn.relu)

    hidden2_dropout = tf.nn.dropout(hidden2, keep_prob)

    logits = nn_layer(
        hidden2_dropout,
        n_outputs,
        name='logits')

  return X, y, is_train, keep_prob, logits

def nn_mnist_model_small_50(graph):
  with graph.as_default():
    is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
  n_inputs = 28*28
  n_hidden1 = 50
  n_hidden2 = 50
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

    logits = nn_layer(
        hidden2,
        n_outputs,
        name='logits')

  return X, y, is_train, logits

def nn_mnist_model_small_500(graph):
  with graph.as_default():
    is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
  n_inputs = 28*28
  n_hidden1 = 500
  n_hidden2 = 500
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

    logits = nn_layer(
        hidden2,
        n_outputs,
        name='logits')

  return X, y, is_train, logits

def nn_mnist_model_small_with_dropout(graph):
  with graph.as_default():
    is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
  n_inputs = 28*28
  n_hidden1 = 5
  n_hidden2 = 5
  n_outputs = 10
  with graph.as_default():
    with tf.name_scope('Inputs'):
      with tf.name_scope('X'):
        X = tf.placeholder(DTYPE, shape=(None, n_inputs), name='X') # pylint: disable=invalid-name
      with tf.name_scope('y'):
        y = tf.placeholder(tf.int64, shape=(None), name='y') # pylint: disable=invalid-name

    keep_prob = tf.placeholder(DTYPE, name='keep_prob')

    hidden1 = nn_layer(
        X,
        n_hidden1,
        name='hidden1',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden1_dropout = tf.nn.dropout(hidden1, keep_prob)

    hidden2 = nn_layer(
        hidden1_dropout,
        n_hidden2,
        name='hidden2',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden2_dropout = tf.nn.dropout(hidden2, keep_prob)

    logits = nn_layer(
        hidden2,
        n_outputs,
        name='logits')

  return X, y, is_train, keep_prob, logits





def nn_mnist_model(graph):
  """Creates model for NN mnist.

  Returns:
    logits
  """
  with graph.as_default():
    is_train = tf.placeholder(tf.bool, shape=(), name='is_train')

  n_inputs = 28*28
  n_hidden1 = 300
  n_hidden2 = 100
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

    logits = nn_layer(
        hidden2,
        n_outputs,
        name='logits')

  return X, y, is_train, logits


def nn_mnist_model_dropout(graph): # pylint: disable=too-many-locals
  """Creates model for NN mnist.

  Returns:
    logits
  """
  n_inputs = 28*28
  n_hidden1 = 1024
  n_hidden2 = 1024
  n_hidden3 = 2048
  n_outputs = 10
  with graph.as_default():
    is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
  with graph.as_default():
    with tf.name_scope('Inputs'):
      with tf.name_scope('X'):
        X = tf.placeholder(DTYPE, shape=(None, n_inputs), name='X') # pylint: disable=invalid-name
      with tf.name_scope('y'):
        y = tf.placeholder(tf.int64, shape=(None), name='y') # pylint: disable=invalid-name
    #with tf.name_scope('NN'):

    keep_prob = tf.placeholder(DTYPE, name='keep_prob')

    hidden1 = nn_layer(
        X,
        n_hidden1,
        name='hidden1',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden1_dropout = tf.nn.dropout(hidden1, keep_prob)

    hidden2 = nn_layer(
        hidden1_dropout,
        n_hidden2,
        name='hidden2',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden2_dropout = tf.nn.dropout(hidden2, keep_prob)

    hidden3 = nn_layer(
        hidden2_dropout,
        n_hidden3,
        name='hidden3',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden3_dropout = tf.nn.dropout(hidden3, keep_prob)

    logits = nn_layer(
        hidden3_dropout,
        n_outputs,
        name='logits')
  return X, y, is_train, keep_prob, logits
