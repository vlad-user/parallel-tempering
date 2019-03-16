import tensorflow as tf
import numpy as np

from simulator.graph.device_placer import _gpu_device_name
from simulator.simulator_utils import DTYPE

DEFAULT_INITIALIZER = tf.contrib.layers.xavier_initializer()

def flatten(tensor, name='flatten'):
  shape = tensor.get_shape().as_list()[1:]
  len_ = int(np.prod(np.array(shape)))
  reshaped = tf.reshape(tensor, shape=[-1, len_])
  return reshaped


def nn_layer(X, n_neurons, name, activation=None, dtype=None): # pylint: disable=invalid-name
  """Creates NN layer.

  Creates NN layer with W's initialized as truncated normal and
  b's as zeros.

  Args:
    X: Input tensor.
    n_neurons: An integer. Number of neurons in the layer.
    name: A string. Name of the layer.
    activation: Activation function. Optional.

  """
  dtype = (DTYPE if dtype is None else dtype)

  with tf.name_scope(name):
    # dimension of each x in X
    n_inputs = int(X.get_shape()[1])

    stddev = 2.0 / np.sqrt(n_inputs)
    init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev, dtype=dtype)

    with tf.device(_gpu_device_name(0)):
      W = tf.Variable(init, name='W', dtype=dtype) # pylint: disable=invalid-name

      b = tf.Variable(tf.zeros([n_neurons], dtype=dtype), name='b', dtype=dtype) # pylint: disable=invalid-name

      Z = tf.matmul(X, W) + b # pylint: disable=invalid-name

      if activation is not None: # pylint: disable=no-else-return
        return activation(Z)
      else:
        return Z



def resnetblock(inputs, n_channels, activations=[None, None],
                name='resnet'):
  """Creates a residual building block.
  
  [1] https://arxiv.org/pdf/1512.03385.pdf

  Args:
    inputs: A tensor input for a resnet block.
    n_channels: A number of channels (feature maps) of each of the
      internal convolutional layers within a resnet block.
    activations: A list of activations function to apply. If activation
      shouldn't be applied, the corresponding value should be `None`.
    name: A name for the resnet block scope.

  Returns:
    A tensor `F(x) + x` as in [1] (Figure 2).
  """

  in_channels = inputs.get_shape().as_list()[-1]

  if in_channels * 2 == n_channels:
    increase_dimensions = True
    stride = 2
  else:
    increase_dimensions = False
    stride = 1
  print('increase_dimensions:', increase_dimensions)
  with tf.name_scope(name):
    x = inputs

    x = tf.layers.conv2d(x,
                         filters=n_channels,
                         kernel_size=3,
                         padding='SAME',
                         activation=activations[0],
                         name='conv_0')
    x = tf.layers.conv2d(x,
                         filters=n_channels,
                         kernel_size=3,
                         padding='SAME',
                         activation=activations[1],
                         name='conv_1')

    if increase_dimensions:
      pool = tf.nn.avg_pool(inputs, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='VALID')
      F = tf.pad(pool, [[0, 0], [0, 0], [0, 0], [in_channels//2, in_channels//2]])

    else:
      F = inputs
    print('F', F.get_shape().as_list())
    print('x', x.get_shape().as_list())
    return F + x
