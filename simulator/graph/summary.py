import os
import pickle
import sys

import tensorflow as tf

from simulator.exceptions import InvalidDatasetTypeError
from simulator.graph.device_placer import _gpu_device_name


FNAME_PREFIX = 'summary_'
FNAME_SUFFIX = '.log'

class Summary:
  """Helper class for storing training log."""
  def __init__(self, name, optimizers, losses, simulation_num, hessian=False, diffusion_ops=None):
    self._n_replicas = len(optimizers.keys())
    self._name = name
    self.simulation_num = simulation_num
    dirname = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    dirname = os.path.join(dirname, 'summaries')
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    self._dirname = os.path.join(dirname, name)
    if not os.path.exists(self._dirname):
      os.makedirs(self._dirname)

    filename = self._create_log_fname(simulation_num=simulation_num)

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
    self._replica_accepts = {i:[] for i in range(self._n_replicas)}
    self._moa_train_steps = []
    self._moa_test_steps = []
    self._moa_valid_steps = []
    self._moa_train_loss = []
    self._moa_test_loss = []
    self._moa_valid_loss = []
    self._moa_train_err = []
    self._moa_test_err = []
    self._moa_valid_err = []
    self._moa_weights = {i:[] for i in range(self._n_replicas)}
    if diffusion_ops is None:
      self._diffusion_ops = self._create_diffusion_ops(optimizers)
    else:
      self._diffusion_ops = diffusion_ops
    if hessian:
      raise NotImplementedError('Currently no need for this.')
      self._hessian_dirname = os.path.join(self._dirname, 'hessian')
      if not os.path.exists(self._hessian_dirname):
        os.makedirs(self._hessian_dirname)
      self._hess_eigenval_ops = self._create_hessian_eigenvalues_ops(optimizers,
                                                                     losses)

    self._epoch = 0

  def _create_diffusion_ops_old(self, optimizers):
    """Creates ops for for diffused l2 distance in parameter space."""
    with tf.name_scope('diffusion'):
      curr_vars = {
          i:sorted(optimizers[i].trainable_variables, key=lambda x: x.name)
              for i in range(self._n_replicas)
      }
      init_vars = {
          i:[tf.Variable(v.initialized_value(), trainable=False) for v in curr_vars[i]]
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
    return _diffusion_ops

  def _create_diffusion_ops(self, optimizers):
    """Creates ops for for diffused l2 distance in parameter space."""
    curr_vars = {}
    init_vars = {}
    curr_vars_reshaped = {}
    init_vars_reshaped = {}
    curr_vars_concat = {}
    init_vars_concat = {}
    _diffusion_ops = {}

    with tf.name_scope('diffusion'):
      for i in range(self._n_replicas):
        with tf.device(_gpu_device_name(i)):
          curr_vars[i] = sorted(optimizers[i].trainable_variables,
                                key=lambda x: x.name)
          init_vars[i] = [tf.Variable(v.initialized_value(), trainable=False)
                          for v in curr_vars[i]]

          curr_vars_reshaped[i] = [tf.reshape(v, [-1]) for v in curr_vars[i]]

          init_vars_reshaped[i] = [tf.reshape(v, [-1]) for v in init_vars[i]]

          curr_vars_concat[i] = tf.concat(curr_vars_reshaped[i], axis=0)
          init_vars_concat[i] = tf.concat(init_vars_reshaped[i], axis=0)

          _diffusion_ops[i] = tf.norm(curr_vars_concat[i] - init_vars_concat[i])

    return _diffusion_ops

  def _create_hessian_eigenvalues_ops(self, optimizers, losses):
    hessians = {}
    result = {}
    def _compute_hessian(grads, vars_):
      mat = []

      for g in grads:
        derv = []
        derv = [tf.gradients(g, v)[0] for v in vars_]
        derv = [tf.reshape(d, [-1]) for d in derv]
        concatinated = tf.concat(derv, axis=0)
              

        
        mat.append(concatinated)


      return tf.stack(mat)

    for i in range(self._n_replicas):
      with tf.device(_gpu_device_name(i)):
        grads = optimizers[i]._grads
        vars_ = optimizers[i].trainable_variables
        with grads[0].graph.as_default():
          with tf.name_scope('hessian'):
            # This implementation doesn't work with 
            # `sparse_softmax_cross_entropy_with_logits()`s because
            # implementation has fused interaction with `tf.gradients()`.
            # Maybe will work in future version.
            # hessian = tf.hessians(losses[i],
            #                       vars_,
            #                       colocate_gradients_with_ops=True)
            hessian = _compute_hessian(grads, vars_)
            

          with tf.name_scope('eigenvalues'):
            evals, evects = tf.linalg.eigh(hessian, name='eigenvalues')

          result[i] = evals

    return result

  def store_hessian_eigenvals(self, eigenvals):
    picklename = str(self._train_steps[-1]) + '_' + str(self.simulation_num) + '.pkl'
    picklename = os.path.join(self._hessian_dirname, picklename)
    with open(picklename, 'wb') as fo:
      pickle.dump(eigenvals, fo, protocol=pickle.HIGHEST_PROTOCOL)

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

  def get_hessian_eigenvals_ops(self):
    ops = [self._hess_eigenval_ops[i] for i in range(self._n_replicas)]
    return [x for x in y for y in ops]
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
    assert len(self._train_loss[0]) == len(self._diffusion_vals[0])
    assert len(self._train_loss[0]) == len(self._train_err[0])
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
        'accepts': self._replica_accepts,
        'latest_epoch': self._epoch,
        'moa_train_loss': self._moa_train_loss,
        'moa_test_loss': self._moa_test_loss,
        'moa_validation_loss': self._moa_valid_loss,
        'moa_train_error': self._moa_train_err,
        'moa_test_error': self._moa_test_err,
        'moa_validation_error': self._moa_valid_err,
        'moa_train_steps': self._moa_train_steps,
        'moa_test_steps': self._moa_test_steps,
        'moa_validation_steps': self._moa_valid_steps,
        'moa_weights': self._moa_weights
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

  def _create_log_fname(self, simulation_num):

    fname = FNAME_PREFIX + str(simulation_num) + FNAME_SUFFIX
    files = [f for f in os.listdir(self.get_dirname()) if FNAME_PREFIX in f]
    if fname not in files:
      return fname
    else:
      files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    last_file = files[-1]
    last_sim_num = int(last_file.split('_')[1].split('.')[0])

    return FNAME_PREFIX + str(last_sim_num + 1) + FNAME_SUFFIX
