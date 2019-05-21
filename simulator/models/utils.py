import tensorflow as tf

from simulator.simulator_utils import DTYPE
from simulator.models.helpers import DEFAULT_INITIALIZER

def average_pooling_with_weights(value, ksize, strides, padding='VALID', activation=None):
  """Average pooling with learnable coefficients.
  
  This function defines an average pooling as specified in [1]. The
  arguments are the same as for the `tf.nn.avg_pool()`. This function
  multiplies the result of the average pooling by a learnable
  coefficient (one per map) and adds a learnable bias term (one
  per map) and applies activation function (if not `None`).

  [1] “Gradient-Based Learning Applied to Document Recognition”,
      Y. LeCun, L. Bottou, Y. Bengio and P. Haffner (1998).
  """
  res = tf.nn.avg_pool(value, ksize, strides, padding)
  shape = value.get_shape().as_list()[-1:]
  weights = tf.Variable(DEFAULT_INITIALIZER(shape, dtype=DTYPE))

  bias = tf.Variable(tf.zeros(shape, dtype=DTYPE))
  res = res * weights + bias
  if activation is not None:
    res = activation(res)
  return res

def rbf_euclidean_layer(inputs, units, activation=None):
  """Defines RBF layer.
  
  This function defines a RBF layer as specified in [1].

  [1] “Gradient-Based Learning Applied to Document Recognition”,
      Y. LeCun, L. Bottou, Y. Bengio and P. Haffner (1998).
  Args:
    inputs: Input tensor.
    units: A number of units.
    activation: An activation to apply.
  """
  inputdim = inputs.get_shape().as_list()[1]
  shape = (inputdim, units)
  weights = tf.Variable(DEFAULT_INITIALIZER(shape, dtype=DTYPE))
  res = tf.reduce_sum(tf.square(inputs[..., None] - weights), axis=1)
  if activation is not None:
    res = activation(res)
  return res

def apply_gaussian_noise(inputs, istrain, stddev):

  def true_fn(x, istrain=istrain, stddev=stddev):
    res = x + tf.random.normal(shape=tf.shape(x), stddev=stddev)
    return res

  return tf.cond(istrain,
                 true_fn=lambda: true_fn(inputs),
                 false_fn=lambda: tf.identity(inputs))

def augment_images(inputs, istrain):

  def true_fn(x):
    with tf.device('cpu:0'):
      maybe_flipped = tf.image.random_flip_left_right(x)
      padded = tf.pad(maybe_flipped, [[0, 0], [4, 4], [4, 4], [0, 0]])
      cropped = tf.image.random_crop(padded, size=tf.shape(x))
    return cropped

  return tf.cond(istrain,
                 true_fn=lambda: true_fn(inputs),
                 false_fn=lambda: tf.identity(inputs))