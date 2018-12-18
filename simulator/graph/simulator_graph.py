"""Classes and functions that manipulate tensorflow graphs."""
import random
import atexit

import tensorflow as tf
import numpy as np

from simulator.graph.graph_duplicator import copy_and_duplicate
from simulator.graph.optimizers import GDOptimizer
from simulator.graph.optimizers import NormalNoiseGDOptimizer
from simulator.graph.optimizers import GDLDOptimizer
from simulator.graph.optimizers import RMSPropOptimizer
from simulator.graph.summary import Summary
from simulator.exceptions import InvalidDatasetTypeError
from simulator.exceptions import InvalidModelFuncError
from simulator.exceptions import InvalidTensorTypeError
from simulator.exceptions import InvalidNoiseTypeError
from simulator.simulator_utils import DTYPE

class SimulatorGraph: # pylint: disable=too-many-instance-attributes
  """Defines a dataflow graph with duplicated ensembles.

  This object stores all copies of the systems at different
  temperatures and provides an API for performing the
  exchanges between two ensembles and storing the summary
  values. It is used to train models in the
  Parallel Tempering framework.
  """

  def __init__(self, model, learning_rate, noise_list, name, # pylint: disable=too-many-locals, too-many-branches, too-many-arguments, too-many-statements
               noise_type='random_normal', simulation_num=None, 
               loss_func_name='cross_entropy', proba_coeff=1.0,
               rmsprop_decay=0.9, rmsprop_momentum=0.0,
               rmsprop_epsilon=1e-10):
    """Instantiates a new SimulatorGraph object.

    Args:
      model: A function that creates inference model (e.g.
        see simulation.models.nn_mnist_model)
      learning_rate: Learning rate for optimizer
      noise_list: A list (not np.array!) for noise/temperatures/dropout
        values. In case of dropout (dropout_rmsprop, dropout_gd), noise_list
        represents the values of KEEPING the neurons, and NOT the probability
        of excluding the neurons.
      noise_type: A string specifying the noise type and optimizer to apply.
        Possible values could be seen at
        simulator.graph.graph_builder.SimulatorGraph.__noise_types
      batch_size: Batch Size
      n_epochs: Number of epochs for each simulation
      name: A name of the simulation. Specifies the a folder name through which
          a summary files can be later accessed.
      simulation_num: Specifies the simulation number that is currently in
        progress. It is relevant when we simulating the same simulation
        multiple times. In this case, each simulation is stored in
        the location: 'summaries/name/simulation_num'.
      loss_func_name: A function which we want to optimize. Currently,
        only cross_entropy and stun (stochastic tunneling) are
        supported.
      proba_coeff: The coeffecient is used in calculation of probability
        of swaps. Specifically, we have
        P(accept_swap) = exp(proba_coeff*(beta_1-beta_2)(E_1-E_2))
      rmsprop_decay: Used in
        simulator.graph.optimizers.RMSPropOptimizer
        for noise type 'dropout_rmsprop'. This value is ignored for
        other noise_types.
      rmsprop_momentum: Used in
        simulator.graph.optimizers.RMSPropOptimizer
        for noise type 'dropout_rmsprop'. This value is ignored for
        other noise_types.
      rmsprop_epsilon: Used in
        simulator.graph.optimizers.RMSPropOptimizer
        for noise type 'dropout_rmsprop'. This value is ignored for
        other noise_types.
    """

    self.__noise_types = ['random_normal',
                          'langevin',
                          'dropout',
                          'dropout_rmsprop',
                          'dropout_gd'] # possible noise types

    if not isinstance(noise_list, list) or not noise_list:
      raise ValueError("Invalid `noise_list`. Should be non-empty python list.")

    self._model = model
    self._learning_rate = learning_rate
    self._n_replicas = len(noise_list)
    self._noise_type = noise_type
    self._name = name
    self._graph = tf.Graph()
    self._noise_list = sorted(noise_list)
    self._simulation_num = 0 if simulation_num is None else simulation_num
    self._loss_func_name = loss_func_name
    self._proba_coeff = proba_coeff
    self._swap_attempts = 0
    self._swap_successes = 0
    self._accept_ratio = 0

    # create graph with duplicated ensembles based on the provided
    # model function and noise type
    res = []
    try:
      res = self._model(tf.Graph())
      if (len(res) == 3 and
          noise_type in ['random_normal', 'langevin']):
        X, y, logits = res # pylint: disable=invalid-name
        self.X, self.y, logits_list = copy_and_duplicate(X, y, logits, # pylint: disable=invalid-name
                                                         self._n_replicas,
                                                         self._graph)

        # _noise_plcholders will be used to store noise vals for summaries
        with self._graph.as_default(): # pylint: disable=not-context-manager
          self._noise_plcholders = {i:tf.placeholder(DTYPE, shape=[])
                                    for i in range(self._n_replicas)}

        # curr_noise_dict stores {replica_id:current noise stddev VALUE}
        self._curr_noise_dict = {i:n for i, n in enumerate(self._noise_list)}

      elif (len(res) == 4
            and noise_type in ['dropout', 'dropout_rmsprop', 'dropout_gd']):
        X, y, prob_placeholder, logits = res # pylint: disable=invalid-name
        self.X, self.y, probs, logits_list = copy_and_duplicate(
            X, y, logits, self._n_replicas, self._graph,
            prob_placeholder)

        # _noise_plcholders stores dropout plcholders: {replica_id:plcholder}
        # it is used also to store summaries
        self._noise_plcholders = {i:p for i, p in enumerate(probs)}

        # in case of noise_type == dropout, _curr_noise_dict stores
        # probabilities for keeping optimization parameters
        # (W's and b's): {replica_id:keep_proba}
        self._curr_noise_dict = {
            i:n
            for i, n in enumerate(sorted(self._noise_list, reverse=True))}

      elif noise_type not in self.__noise_types:
        raise InvalidNoiseTypeError(noise_type, self.__noise_types)

      else:
        raise InvalidModelFuncError(len(res), self._noise_type)

    except Exception as exc:
      
      raise Exception("Problem with inference model function.") from exc

    # from here, everything that goes after logits is created
    self._loss_dict = {}
    self._error_dict = {}
    self._optimizer_dict = {}

    # set loss function and optimizer
    with self._graph.as_default(): # pylint: disable=not-context-manager
      for i in range(self._n_replicas):

        with tf.name_scope('Metrics' + str(i)):
          if self._loss_func_name in ['cross_entropy', 'crossentropy']:
            loss_func = self._cross_entropy_loss
          elif self._loss_func_name in ['stun']:
            loss_func = self._stun_loss
          else:
            raise ValueError('Invalid loss function name.',
                             'Available functions are: \
                             cross_entropy/stun,',
                             'But given:', self._loss_func_name)
          
          self._loss_dict[i] = loss_func(
              self.y, logits_list[i])

          self._error_dict[i] = self._error(
              self.y, logits_list[i])

        with tf.name_scope('Optimizer_' + str(i)):

          if noise_type.lower() == 'random_normal':
            optimizer = NormalNoiseGDOptimizer(
                self._learning_rate, i, self._noise_list)

          elif noise_type.lower() == 'langevin':
            optimizer = GDLDOptimizer(
                self._learning_rate, i, self._noise_list)

          elif noise_type.lower() == 'dropout_rmsprop':
            optimizer = RMSPropOptimizer(
                self._learning_rate, i, self._noise_list,
                decay=rmsprop_decay, momentum=rmsprop_momentum,
                epsilon=rmsprop_epsilon)

          elif noise_type.lower() in ['dropout_gd', 'dropout']:
            optimizer = GDOptimizer(
                self._learning_rate, i, self._noise_list)

          else:
            raise InvalidNoiseTypeError(noise_type, self.__noise_types)
          optimizer.minimize(self._loss_dict[i])
          self._optimizer_dict[i] = optimizer


      self._summary = Summary(self._name,
                              self._optimizer_dict,
                              simulation_num)
      self.variable_initializer = tf.global_variables_initializer()
      # flush summary on exit
      atexit.register(self._summary.flush_summary)
      
  def flush_summary(self):
    self._summary.flush_summary()

  def create_feed_dict(self, X_batch, y_batch, dataset_type='train'): # pylint: disable=invalid-name
    """Creates feed_dict for session run.

    Args:
      `X_batch`: input X training batch
      `y_batch`: input y training batch
      `dataset_type`: 'train', 'test' or 'validation'

    Returns:
      A dictionary to feed into session's run.
      If `dataset_type` == 'train', adds to feed_dict placeholders
      to store noise (for summary).
      If `dataset_type` == 'validation'/'test', then doesn't add
      this placeholder (since we don't add noise for test or
      validation).
      If noise_type is 'dropout' and dataset_type is 'train',
      adds values for keeping parameters during optimization
      (placeholders `keep_prob` for each replica).

    Raises:
      InvalidDatasetTypeError: if incorrect `dataset_type`.
    """

    feed_dict = {self.X:X_batch, self.y:y_batch}

    if dataset_type == 'test' and 'dropout' in self._noise_type:
      dict_ = {self._noise_plcholders[i]:1.0
               for i in range(self._n_replicas)}

    elif dataset_type == 'validation' and 'dropout' in self._noise_type:
      dict_ = {self._noise_plcholders[i]:1.0
               for i in range(self._n_replicas)}

    elif dataset_type == 'train' and 'dropout' in self._noise_type:
      dict_ = {self._noise_plcholders[i]:self._curr_noise_dict[i]
               for i in range(self._n_replicas)}

    elif dataset_type not in ['train', 'test', 'validation']:
      raise InvalidDatasetTypeError()
    
    else:
      dict_ = {}

    feed_dict.update(dict_)

    return feed_dict

  def get_train_ops(self, dataset_type='train', include_diffusion=False):
    """Returns train ops for session's run.

    The returned list should be used as:
    # evaluated = sess.run(get_train_ops(), feed_dict=...)

    Args:
      dataset_type: One of 'train'/'test'/'validation'
      include_diffusion: If True, includes ops for calculating diffusion.

    Returns:
      train_ops for session run.

    Raises:
      InvalidDatasetTypeError: if incorrect dataset_type.
    """

    if dataset_type not in ['train', 'test', 'validation']:
      raise InvalidDatasetTypeError()

    loss = [self._loss_dict[i]
            for i in range(self._n_replicas)]
    error = [self._error_dict[i]
                     for i in range(self._n_replicas)]
    res = loss + error
    
    if include_diffusion:
      res += self._summary.get_diffusion_ops()
      res += self._summary.get_grad_norm_ops()
      res += self._summary.get_norm_ops()

    if dataset_type == 'train':
      res = res + [self._optimizer_dict[i].get_train_op()
                   for i in range(self._n_replicas)]

    return res

  def add_summary(self, evaluated, step, epoch=None, dataset_type='train'):
    """Adds summary evaluated summary.

    ### Usage:

    ```python
    # Suppose g is a SimulatorGraph object and step is the value
    # that is incremented after each mini-batch.
    # Evaluate train data and store computed values:
    evaluated = sess.run(g.get_train_ops(dataset_type='train'))
    g.add_summary(evaluated, step, dataset_type='train')
    ```

    Args:
      evaluated: A list returned by `sess.run(get_train_ops())`
      step: A value that is incremented after each mini-batch.
      epoch: (Optional) A current epoch that allows accuratly show
        the progress during training via SummaryExtractor class.
      dataset_type: One of 'train'/'test'/'validation'.

    """

    def has_diffusion_ops():
      """Returns True if evaluated contains diffusion ops."""
      return ((dataset_type in ['test', 'validation']
              and len(evaluated) == 5*self._n_replicas)
            or (dataset_type in ['train']
              and len(evaluated) == 6*self._n_replicas))

    error = self.extract_evaluated_tensors(evaluated, 'error')
    loss = self.extract_evaluated_tensors(evaluated, 'loss')

    if dataset_type == 'train':
      loss_dict = self._summary._train_loss
      error_dict = self._summary._train_err
      steps = self._summary._train_steps
      self._summary.add_noise_vals(self._curr_noise_dict)
    elif dataset_type == 'test':
      loss_dict = self._summary._test_loss
      error_dict = self._summary._test_err
      steps = self._summary._test_steps
    elif dataset_type == 'validation':
      loss_dict = self._summary._valid_loss
      error_dict = self._summary._valid_err
      steps = self._summary._valid_steps
    else:
      raise InvalidDatasetTypeError()

    for i in range(self._n_replicas):
      loss_dict[i].append(loss[i])
      error_dict[i].append(error[i])
    steps.append(step)

    # add diffusion if exists in the `evaluated` list
    if has_diffusion_ops():
      diffusion = self.extract_evaluated_tensors(evaluated, 'diffusion') 
      for i in range(self._n_replicas):
        self._summary._diffusion_vals[i].append(diffusion[i])
        self._summary._grads_vals[i].append(diffusion[i+self._n_replicas])
        self._summary._weight_norm_vals[i].append(diffusion[i+2*self._n_replicas])

    if epoch is not None:
      self._summary._epoch = epoch



  def extract_evaluated_tensors(self, evaluated, tensor_type):
    """Extracts tensors from a list of tensors evaluated by tf.Session.

    ### Usage:

    ```python
    # Suppose g is a SimulatorGraph object.
    # Run and print cross entropy loss vals for each replica for test data:
    evaluated = sess.run(g.get_train_ops(dataset_type='test'))
    loss_vals = g.extract_evaluated_tensors(evaluated,
      tensor_type='loss')
    print(loss_vals)
    ```

    Args:
      evaluated: A list returned by sess.run(get_train_ops())
      tensor_type: One of 'loss'/'error'/'diffusion'

    Returns:
      A list of specified (by `tensor_type`) tensor values.

    Raises:
      InvlaidLossFuncError: Incorrect `tensor_type` value.
      """

    if tensor_type == 'loss': # pylint: disable=no-else-return
      return evaluated[:self._n_replicas]

    elif tensor_type == 'error':
      return evaluated[self._n_replicas:self._n_replicas*2]

    elif tensor_type == 'diffusion': # includes diffusion, grad_norms, weight_norms
      return evaluated[self._n_replicas*2:self._n_replicas*5]

    else:
      raise InvalidTensorTypeError()

  def swap_replicas(self, evaluated): # pylint: disable=too-many-locals
    """Swaps between replicas.

    Swaps according to:
      1. Uniformly randomly select a pair of adjacent temperatures
        1/beta_i and 1/beta_i+1, for which swap move is proposed.
      2. Accepts swap with probability:
          min{1, exp(proba_coeff*(beta_i-beta_i+1)*(loss_i-loss_i+1)}
      3. Updates the acceptance ratio for the proposed swap.

    Args:
      evaluated: a list returned by sess.run(get_train_ops())

    Raises:
      ValueError: if invalid `surface_view`.
    """
    random_pair = random.choice(range(self._n_replicas - 1)) # pair number

    beta = [self._curr_noise_dict[x] for x in range(self._n_replicas)]
    beta_id = [(b, i) for i, b in enumerate(beta)]
    beta_id.sort(key=lambda x: x[0], reverse=True)

    i = beta_id[random_pair][1]
    j = beta_id[random_pair+1][1]

    loss_list = self.extract_evaluated_tensors(evaluated, 'loss')

    li, lj = loss_list[i], loss_list[j]
    proba = np.exp(self._proba_coeff*(li-lj)*(beta[i]-beta[j]))

    if np.random.uniform() < proba:
      self._curr_noise_dict[i] = beta[j]
      self._curr_noise_dict[j] = beta[i]
      self._optimizer_dict[i].set_train_route(j)
      self._optimizer_dict[j].set_train_route(i)

      self._swap_attempts += 1
      self._swap_successes += 1
      accept_pair = [(i, 1), (j, 1)]
    else:
      self._swap_attempts += 1
      accept_pair = [(i, 0), (j, 0)]

    for p in accept_pair:
      self._summary._replica_accepts[p[0]].append(p[1])

    self._accept_ratio = self._swap_successes / self._swap_attempts


  def get_tf_graph(self):
    """Returns tensorflow graph."""
    return self._graph

  def get_summary_dirname(self):
    """Returns a directory name where summary is stored."""
    return self._summary.get_dirname()

  def get_accept_ratio(self):
    """Returns current accept ratio of all replicas."""
    return self._accept_ratio

  def _store_tf_graph(self, path):
    """Stores tensorflow graph."""
    tf.summary.FileWriter(path, self._graph).close()

  def _cross_entropy_loss(self, y, logits, clip_value_max=2000.0): # pylint: disable=invalid-name, no-self-use
    """Cross entropy on cpu"""
    with tf.name_scope('cross_entropy'):
      with tf.device('/cpu:0'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits)
        loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
        if clip_value_max is not None:
          loss = tf.clip_by_value(loss, 0.0, clip_value_max)
    return loss

  def _error(self, y, logits): # pylint: disable=invalid-name, no-self-use
    """0-1 loss"""
    with tf.name_scope('error'):
      with tf.device('/cpu:0'):
        # input arg `perdictions` to tf.nn.in_top_k must be of type tf.float32
        y_pred = tf.nn.in_top_k(predictions=tf.cast(logits, tf.float32),
                                targets=y,
                                k=1)
        error = 1.0 - tf.reduce_mean(tf.cast(x=y_pred, dtype=DTYPE),
                                     name='error')
    return error

  def _stun_loss(self, loss, gamma=1): # pylint: disable=no-self-use
    """Stochastic tunnelling loss."""
    with tf.name_scope('stun'):
      with tf.device('/cpu:0'):
        stun = 1 - tf.exp(-gamma*loss) # pylint: disable=no-member

    return stun

  