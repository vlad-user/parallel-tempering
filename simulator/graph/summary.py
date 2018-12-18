import os
import pickle
import sys

import tensorflow as tf

from simulator.exceptions import InvalidDatasetTypeError

class Summary:
  """Helper class for storing training log."""
  def __init__(self, name, optimizers, simulation_num):
    self._n_replicas = len(optimizers.keys())
    self._name = name
    dirname = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    dirname = os.path.join(dirname, 'summaries')
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    self._dirname = os.path.join(dirname, name)
    if not os.path.exists(self._dirname):
      os.makedirs(self._dirname)
    filename = 'summary_' + str(simulation_num) + '.log'
    self._logfile_name = os.path.join(self._dirname, filename)
    

    self._train_loss = {i:[] for i in range(self._n_replicas)}
    self._train_err = {i:[] for i in range(self._n_replicas)}
    self._train_steps = []
    self._test_loss = {i:[] for i in range(self._n_replicas)}
    self._test_err = {i:[] for i in range(self._n_replicas)}
    self._test_steps = []
    self._valid_loss = {i:[] for i in range(self._n_replicas)}
    self._valid_err = {i:[] for i in range(self._n_replicas)}
    self._valid_steps = []
    self._train_noise_vals = {i:[] for i in range(self._n_replicas)}
    self._diffusion_vals = {i:[] for i in range(self._n_replicas)}
    self._grads_vals = {i:[] for i in range(self._n_replicas)}
    self._weight_norm_vals = {i:[] for i in range(self._n_replicas)}

    self._replica_accepts = {i:[] for i in range(self._n_replicas)}

    self._diffusion_ops, self._norm_ops = self._create_diffusion_ops(optimizers)

    self._grad_norm_ops = self._create_gradient_ops(optimizers)

    self._epoch = 0

  def _create_diffusion_ops(self, optimizers):
    """Creates ops for for diffused l2 distance in parameter space."""
    with tf.name_scope('diffusion'):
      curr_vars = {
          i:sorted(optimizers[i].trainable_variables, key=lambda x: x.name)
              for i in range(self._n_replicas)
      }
      init_vars = {
          i:[tf.Variable(v.initialized_value()) for v in curr_vars[i]]
              for i in range(self._n_replicas)
      }
      curr_vars_reshaped = {
          i:[tf.reshape(v, [-1]) for v in curr_vars[i]]
              for i in range(self._n_replicas)
      }
      init_vars_reshaped = {
          i:[tf.reshape(v, [-1]) for v in init_vars[i]]
              for i in range(self._n_replicas)
      }
      curr_vars_concat = {
          i:tf.concat(curr_vars_reshaped[i], axis=0)
              for i in range(self._n_replicas)
      }
      init_vars_concat = {
          i:tf.concat(init_vars_reshaped[i], axis=0)
              for i in range(self._n_replicas)
      }
      _diffusion_ops = {
          i:tf.norm(curr_vars_concat[i] - init_vars_concat[i])
              for i in range(self._n_replicas)
      }
      _norm_ops = {
          i:tf.norm(curr_vars_concat[i])
              for i in range(self._n_replicas)
      }
    return _diffusion_ops, _norm_ops

  def _create_gradient_ops(self, optimizers):
    with tf.name_scope('gradient_norms'):

      curr_vars = {
          i:sorted(optimizers[i]._grads, key=lambda x: x.name)
              for i in range(self._n_replicas)
      }
      curr_vars_reshaped = {
          i:[tf.reshape(v, [-1]) for v in curr_vars[i]]
              for i in range(self._n_replicas)
      }
      curr_vars_concat = {
          i:tf.concat(curr_vars_reshaped[i], axis=0)
              for i in range(self._n_replicas)
      }
      grad_ops = {
          i:tf.norm(curr_vars_concat[i])
              for i in range(self._n_replicas)
      }

    return grad_ops

  def get_diffusion_ops(self):
    return [self._diffusion_ops[i] for i in range(self._n_replicas)]

  def get_norm_ops(self):
    return [self._norm_ops[i] for i in range(self._n_replicas)]

  def get_grad_norm_ops(self):
    return [self._grad_norm_ops[i] for i in range(self._n_replicas)]

  def add_noise_vals(self, noise_dict):
    """Updates current noise values."""
    for i in range(self._n_replicas):
      self._train_noise_vals[i].append(noise_dict[i])

  def flush_summary(self):
    """Flushes summary log to a disk.

    **WARNING**: On Windows simultaneous writing/reading to a file
    is not supported. Reading a train log file on Windows during
    training may damage the log data.
    """
    log_data = {
        'train_loss': self._train_loss,
        'train_error': self._train_err,
        'train_steps': self._train_steps,
        'test_loss': self._test_loss,
        'test_error': self._test_err,
        'test_steps': self._test_steps,
        'validation_loss': self._valid_loss,
        'validation_error': self._valid_err,
        'validation_steps': self._valid_steps,
        'noise_values': self._train_noise_vals,
        'diffusion': self._diffusion_vals,
        'grad_norms': self._grads_vals,
        'weight_norms': self._weight_norm_vals,
        'accepts': self._replica_accepts,
        'latest_epoch': self._epoch
    }
    if 'win' in sys.platform:
      with open(self._logfile_name, 'wb') as fo:
        pickle.dump(log_data, fo, protocol=pickle.HIGHEST_PROTOCOL)

    else:
      with open(self._logfile_name, 'wb', os.O_NONBLOCK) as fo:
        pickle.dump(log_data, fo, protocol=pickle.HIGHEST_PROTOCOL)

  def get_dirname(self):
    """Returns a full path to the directory where the log is stored."""
    return self._dirname