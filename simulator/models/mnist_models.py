"""NN models for mnist"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib # pylint: disable=no-name-in-module

from simulator.graph.device_placer import _gpu_device_name
from simulator.simulator_utils import DTYPE

def flatten(tensor, name='flatten'):
  shape = tensor.get_shape().as_list()[1:]
  len_ = int(np.prod(np.array(shape)))
  reshaped = tf.reshape(tensor, shape=[-1, len_])
  return reshaped

def nn_layer(X, n_neurons, name, activation=None): # pylint: disable=invalid-name
  """Creates NN layer.

  Creates NN layer with W's initialized as truncated normal and
  b's as zeros.

  Args:
    X: Input tensor.
    n_neurons: An integer. Number of neurons in the layer.
    name: A string. Name of the layer.
    activation: Activation function. Optional.

  """
  with tf.name_scope(name):
    # dimension of each x in X
    n_inputs = int(X.get_shape()[1])

    stddev = 2.0 / np.sqrt(n_inputs)
    init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev, dtype=DTYPE)

    with tf.device(_gpu_device_name(0)):
      W = tf.Variable(init, name='W', dtype=DTYPE) # pylint: disable=invalid-name

      b = tf.Variable(tf.zeros([n_neurons], dtype=DTYPE), name='b', dtype=DTYPE) # pylint: disable=invalid-name

      Z = tf.matmul(X, W) + b # pylint: disable=invalid-name

      if activation is not None: # pylint: disable=no-else-return
        return activation(Z)
      else:
        return Z

def resblock(x_init, channels, use_bias=True, downsample=False, scope='resblock'):
  return 1

############################ WORKING MODELS ############################

def nn_mnist_model_small(graph):
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

  return X, y, logits

def nn_mnist_model_small_with_dropout(graph):
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

  return X, y, keep_prob, logits

def nn_mnist_model_075(graph):
  n_inputs = 28*28
  n_hidden1 = int(300*0.75)
  n_hidden2 = int(100*0.75)
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

  return X, y, logits

def nn_mnist_model_125(graph):
  n_inputs = 28*28
  n_hidden1 = int(300*1.25)
  n_hidden2 = int(100*1.25)
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

  return X, y, logits


def nn_mnist_model(graph):
  """Creates model for NN mnist.

  Returns:
    logits
  """
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

  return X, y, logits

def nn_mnist_model2(graph):
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

def nn_mnist_model2_05(graph):
  """Creates model for NN mnist.

  Returns:
    logits
  """
  n_inputs = 28*28
  n_hidden1 = int(1024*0.5)
  n_hidden2 = int(1024*0.5)
  n_hidden3 = int(2048*0.5)
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
        activation=tf.nn.relu) # pylint: disable=no-member

    logits = nn_layer(
        hidden3,
        n_outputs,
        name='logits')

  return X, y, logits

def nn_mnist_model2_150(graph):
  """Creates model for NN mnist.

  Returns:
    logits
  """
  n_inputs = 28*28
  n_hidden1 = int(1024*1.5)
  n_hidden2 = int(1024*1.5)
  n_hidden3 = int(2048*1.5)
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
        activation=tf.nn.relu) # pylint: disable=no-member

    logits = nn_layer(
        hidden3,
        n_outputs,
        name='logits')

  return X, y, logits




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
  return X, y, keep_prob, logits


def nn_mnist_model_batch_norm(graph):
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

    hidden1_norm = tf.layers.batch_normalization(n_hidden1, name='batch_norm1')

    hidden2 = nn_layer(
        hidden1_norm,
        n_hidden2,
        name='hidden2',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden2_norm = tf.layers.batch_normalization(n_hidden2, name='batch_norm2')

    hidden3 = nn_layer(
        hidden2_norm,
        n_hidden3,
        name='hidden2',
        activation=tf.nn.relu)# pylint: disable=no-member

    hidden3_norm = tf.layers.batch_normalization(n_hidden3, name='batch_norm3')

    logits = nn_layer(
        hidden3_norm,
        n_outputs,
        name='logits')

  return X, y, logits

def resnet_mnist(graph):

  img_size = 28
  c_dim = 1
  label_dim = 10

  with graph.as_default():
    with tf.name_scope('Inputs'):
      with tf.name_scope('X'):
        X = tf.placeholder(DTYPE, shape=(None, width*height), name='X')
        X_reshaped = tf.reshape(x, [None, width , height, -1])

      with tf.name_scope('y'):
        y = tf.placeholder(DTYPE, shape=(None), name='y')

    with tf.name_scope('conv0'):
      net = tf.layers.conv2d(
          X_reshaped, 16, [3,3], strides=1, padding='SAME', use_bias=False,
          kernel_initilizer=tf.contrib.layers.variance_scaling_initializer(
              seed=seed),
          )
      net = tf.layers.batch_normalization(net, training)
