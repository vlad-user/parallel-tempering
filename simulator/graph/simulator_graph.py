"""Classes and functions that manipulate tensorflow graphs."""
import random

import tensorflow as tf
import numpy as np

from simulator.graph.graph_duplicator import copy_and_duplicate
from simulator.graph.optimizers import GDOptimizer
from simulator.graph.optimizers import NormalNoiseGDOptimizer
from simulator.graph.optimizers import GDLDOptimizer
from simulator.graph.optimizers import RMSPropOptimizer
from simulator.graph.optimizers import _get_dependencies
from simulator.graph.summary import Summary
from simulator.exceptions import InvalidDatasetTypeError
from simulator.exceptions import InvalidModelFuncError
from simulator.exceptions import InvalidTensorTypeError
from simulator.exceptions import InvalidNoiseTypeError
from simulator.simulator_utils import DTYPE
from simulator.graph.device_placer import _gpu_device_name

class SimulatorGraph:
  """Defines a dataflow graph with duplicated ensembles.

  This object stores all copies of the systems at different
  temperatures and provides an API for performing the
  exchanges between two ensembles and storing the summary
  values. It is used to train models in the
  Parallel Tempering framework.
  
  ##Notes on `noise_type`s:
  * `l2_regularizer_with_dropout` applies dropout both on hidden
    units and on squared trainable variables.

  # Example:
  ```python
  import tensorflow as tf
  import simulator.models.nn_mnist_model_small as model
  from simulator.graph.simulator_graph import SimulatorGraph
  graph = SimulatorGraph(model=model,
                         learning_rate=0.01,
                         noise_list=[100, 200],
                         name='test_simulator_graph',
                         noise_type='langevin',
                         )
  with graph.get_tf_graph().as_default():
    data = tf.data.Dataset.from_tensor_slices({
        'X': train_data,
        'y': train_labels
    })
    iterator = data.make_initializable_iterator()

  n_epochs = 5
  batch_size = 50
  with graph.get_tf_graph().as_default():
    sess.run(iterator.initializer)
    sess.run(graph.variable_initializer)
    next_batch = iterator.get_next()
    for epoch in range(n_epochs):
      try:
        batch = sess.run(next_batch)
        feed_dict = graph.create_feed_dict(batch['X'], batch['y'])
        evaluated = sess.run(graph.get_train_ops(),
                             feed_dict=feed_dict)
        loss = graph.extract_evaluated_tensors(evaluated, 'loss')
        err = graph.extract_evaluated_tensors(evaluated, 'error')
        print(loss)
        print(err)
        graph.add_summary(evaluated, step, dataset_type='train')
        graph.flush_summary()

      except tf.errors.OutOfRangeError:
        sess.run(iterator.initializer)
        break
    graph.flush_summary()
  ```
  """

  _noise_types = ['weight_noise',
                  'langevin',
                  'dropout',
                  'dropout_rmsprop',
                  'dropout_gd',
                  'gd_no_noise',
                  'l2_regularizer',
                  'l2_regularizer_with_dropout',
                  'input_noise',
                  'learning_rate'] # possible noise types
  _summary_types = ['diffusion_summary',
                    'train_loss_summary',
                    'train_error_summary',
                    'test_loss_summary',
                    'test_error_summary',
                    'validation_loss_summary',
                    'validation_error_summary',
                    'train_steps_summary',
                    'test_steps_summary',
                    'validation_steps_summary',
                    'moa_train_loss_summary',
                    'moa_test_loss_summary',
                    'moa_validation_loss_summary',
                    'moa_train_error_summary',
                    'moa_test_error_summary',
                    'moa_validation_error_summary',
                    'moa_train_steps_summary',
                    'moa_test_steps_summary',
                    'moa_validation_steps_summary',
                    'moa_weights_summary']

  def __init__(self, model, learning_rate, noise_list, name,
               ensembles=None, noise_type='weight_noise',
               simulation_num=None, loss_func_name='cross_entropy',
               proba_coeff=1.0, hessian=False, mode=None,
               moa_lr=None, lambda_=0.01,
               rmsprop_decay=0.9, rmsprop_momentum=0.0,
               rmsprop_epsilon=1e-10):
    """Instantiates a new SimulatorGraph object.

    Args:
      model: A function that creates inference model (e.g.
        see `simulation.models.nn_mnist_model()`).
      learning_rate: Learning rate for optimizer.
      noise_list: A list (not `np.array`!) for
        noise/temperatures/dropout values. In case of dropout
        (dropout_rmsprop, dropout_gd), noise_list represents the values
        of KEEPING the neurons, and NOT the probability of excluding
        the neurons.
      noise_type: A string specifying the noise type and optimizer
        to apply. Possible values could be seen at
        `simulator.graph.graph_builder.SimulatorGraph._noise_types`.
      batch_size: Batch Size.
      n_epochs: Number of epochs for each simulation
      name: A name of the simulation. Specifies the a folder name
        through which a summary files can be later accessed.
      ensembles: A dictionary having as keys strings 'X', 'y',
        'is_train', 'logits_list', 'keep_prob_list' and their
        respective values. Some of the TensorFlow tensors connot
        be copied by `graph_duplicator` module (e.g.
        `tf.layers.batch_normalization`) and it will throw
        `tf.error.FailedPreconditionError`. So all of the replicas
        should be implemented manually and the `ensembles` dictionary
        should contain all the specified values necessary to proceed.
        If the `ensembles` is not `None`, the argument `model` is
        ignored. The lists `logits_list` and `keep_prob_list` must
        contain the values for each replica in ordered form s.t.
        first element in both lists should correspond to the ensemble
        0, the second element should correspond to the ensemble 1 e.t.c.
      simulation_num: Specifies the simulation number that is
        currently in progress. It is relevant when we simulating the
        same simulation multiple times. In this case, each simulation
        is stored in the location: 'summaries/name/simulation_num'.
      loss_func_name: A function which we want to optimize. Currently,
        only `cross_entropy` and `stun` (stochastic tunneling) are
        supported.
      proba_coeff: The coeffecient is used in calculation of
        probability of swaps. Specifically, we have
        `P(accept_swap) = exp(proba_coeff*(beta_1-beta_2)(E_1-E_2))`
      hessian: (Boolean) If `True`, computes Hessian and its
        eigenvalues during each swap step. Default is `False` since
        it is computationally intensive and should be used only for
        small networks. **Not implemented**.
      mode: If `None` (default) will prepare graph for regular parallel
        tempering simulation. If one of `['MOA', 'mixture_of_agents']`,
        in addition to a parallel tempering mode will also train and
        do inference on a weighted outputs of `logits` of all replicas.
        The weight update of such MOA structure is made only on weights
        that `logits` of each replica are multiplied. The updates for
        weights of each replica are made as in usual parallel tempering
        simulation. The learning rate for MOA update is taken from
        `moa_lr` argument.
      moa_lr: A learning rate for `MOA` mode. If `None`, the
        learning rate for replicas is used.
      lambda_: In case of `l2_regularizer_with_dropout`, the constant
        regularization parameter is applied for every replica. For
        other noise types, this value is ignored.
      rmsprop_decay: Used in
        simulator.graph.optimizers.RMSPropOptimizer
        for noise type 'dropout_rmsprop'. This value is ignored for
        other `noise_types`.
      rmsprop_momentum: Used in
        `simulator.graph.optimizers.RMSPropOptimizer`
        for noise type 'dropout_rmsprop'. This value is ignored for
        other `noise_types`.
      rmsprop_epsilon: Used in
        `simulator.graph.optimizers.RMSPropOptimizer`
        for noise type 'dropout_rmsprop'. This value is ignored for
        other `noise_types`.
    """

    if not isinstance(noise_list, list) or not noise_list:
      raise ValueError("Invalid `noise_list`. Should be non-empty Python list.")

    self._model = model
    self._lr = learning_rate
    self._n_replicas = len(noise_list)
    self._noise_type = noise_type
    self._name = name
    self._noise_list = sorted(noise_list)
    self._simulation_num = 0 if simulation_num is None else simulation_num
    self._loss_func_name = loss_func_name
    self._proba_coeff = proba_coeff
    self._swap_attempts = 0
    self._swap_successes = 0
    self._accept_ratio = 0
    self._hessian = hessian
    self._lambda = lambda_
    self._mode = mode
    self._moa_lr = (learning_rate if moa_lr is None else moa_lr)

    # create graph with duplicated ensembles based on the provided
    # model function and noise type
    if ensembles is not None:
      try:
        self.X = ensembles['x']
        self.y = ensembles['y']
        self.is_train = ensembles['is_train']
        logits_list = ensembles['logits_list']
        self.logits_list = logits_list
        self._graph = self.X.graph
        if 'keep_prob_list' in ensembles:
          probs = ensembles['keep_prob_list']
          if noise_type not in ['dropout', 'dropout_rmsprop', 'dropout_gd']:
            raise InvalidNoiseTypeError(noise_type, self._noise_types)
          self._noise_plcholders = {i:p for i, p in enumerate(probs)}
        else:
          with self._graph.as_default():
            self._noise_plcholders = {i:tf.placeholder(DTYPE, shape=[])
                                        for i in range(self._n_replicas)}
      except KeyError as exc:
        raise ValueError("ensemble dictionary not as expected") from exc
    else:
      self._graph = tf.Graph()
      res = []
      try:
        res = self._model(tf.Graph())
        with res[0].graph.as_default():
          self._n_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

        if (len(res) == 4 and
            noise_type in ['weight_noise',
                           'langevin',
                           'gd_no_noise',
                           'l2_regularizer',]):
          X, y, is_train, logits = res

          self.X, self.y, self.is_train, logits_list = copy_and_duplicate(
              X, y, is_train, logits, self._n_replicas, self._graph, self._noise_type)
          self.logits_list = logits_list
          # _noise_plcholders will be used to store noise vals for summaries
          with self._graph.as_default():
            self._noise_plcholders = {i:tf.placeholder(DTYPE, shape=[])
                                      for i in range(self._n_replicas)}

        elif (len(res) == 5
              and noise_type in ['dropout',
                                 'dropout_rmsprop',
                                 'dropout_gd',
                                 'l2_regularizer_with_dropout',
                                 'input_noise',
                                 'learning_rate']):
          X, y, is_train, prob_placeholder, logits = res

          self.X, self.y, self.is_train, probs, logits_list = copy_and_duplicate(
              X, y, is_train, logits, self._n_replicas, self._graph, self._noise_type,
              prob_placeholder)
          self.logits_list = logits_list
          # _noise_plcholders stores dropout plcholders: {replica_id:plcholder}
          # it is used also to store summaries
          self._noise_plcholders = {i:p for i, p in enumerate(probs)}

        elif noise_type not in self._noise_types:
          raise InvalidNoiseTypeError(noise_type, self._noise_types)

        else:
          raise InvalidModelFuncError(len(res), self._noise_type)

      except Exception as exc:
        raise Exception("Problem with inference model function.") from exc

    # curr_noise_dict stores {replica_id:current noise stddev VALUE}
    # If noise type is langevin or weight_noise it will store beta
    # values. In case of noise_type == dropout, _curr_noise_dict stores
    # probabilities for keeping optimization parameters
    self._curr_noise_dict = {i:n for i, n in enumerate(self._noise_list)}

    # from here, everything that goes after logits is created
    self._loss_dict = {}
    self._error_dict = {}
    self._optimizer_dict = {}

    # set loss function and optimizer
    with self._graph.as_default(): # pylint: disable=not-context-manager
      for i in range(self._n_replicas):

        with tf.name_scope('Metrics' + str(i)):
          with tf.device(_gpu_device_name(i)):
            if self._loss_func_name in ['cross_entropy', 'crossentropy']:
              loss_func = self._cross_entropy_loss
            elif self._loss_func_name in ['stun']:
              loss_func = self._stun_loss
            else:
              err_msg = ("Invalid loss function name. "
                         "Available function names are: "
                         "cross_entropy/stun. But given: ")
              err_msg += self._loss_func_name
              raise ValueError(err_msg)
            if self._noise_type == 'l2_regularizer_with_dropout':
              self._loss_dict[i] = loss_func(self.y,
                                             logits_list[i],
                                             regularizer=self._noise_type,
                                             lambda_=self._lambda,
                                             keep_prob=self._self._noise_plcholders[i])
            else:
              self._loss_dict[i] = loss_func(self.y,
                                             logits_list[i],
                                             regularizer=self._noise_type,
                                             lambda_=self._noise_plcholders[i])
          # on cpu
          self._error_dict[i] = self._error(
              self.y, logits_list[i])

        with tf.name_scope('Optimizer_' + str(i)):

          if noise_type.lower() == 'weight_noise':
            optimizer = NormalNoiseGDOptimizer(
                self._lr, i, self._noise_list)

          elif noise_type.lower() == 'langevin':
            optimizer = GDLDOptimizer(
                self._lr, i, self._noise_list)

          elif noise_type.lower() == 'dropout_rmsprop':
            optimizer = RMSPropOptimizer(
                self._lr, i, self._noise_list,
                decay=rmsprop_decay, momentum=rmsprop_momentum,
                epsilon=rmsprop_epsilon)

          elif noise_type.lower() in ['dropout_gd',
                                      'dropout',
                                      'gd_no_noise',
                                      'l2_regularizer',
                                      'input_noise']:
            optimizer = GDOptimizer(
                self._lr, i, self._noise_list)

          elif noise_type.lower() in ['learning_rate']:
            optimizer = GDOptimizer(
                self._noise_plcholders[i], i, self._noise_list)

          else:
            raise InvalidNoiseTypeError(noise_type, self._noise_types)
          optimizer.minimize(self._loss_dict[i])
          self._optimizer_dict[i] = optimizer


      self._summary = Summary(self._name,
                              self._optimizer_dict,
                              self._loss_dict,
                              simulation_num,
                              hessian=self._hessian)

      if self._mode in ['MOA', 'mixture_of_experts', 'moa']:
        self._create_moa_ops()
      self.variable_initializer = tf.global_variables_initializer()

  def _create_moa_ops(self):
    """Creates MOA ops."""
    with self.X.graph.as_default():
      self._moa_weights = tf.Variable(tf.random_normal([self._n_replicas, 1, 1]))
      expanded = [tf.expand_dims(self.logits_list[i], 0) for i in range(self._n_replicas)]
      concated = tf.concat(expanded, 0)
      weighted = self._moa_weights * concated
      moa_logits = tf.reduce_sum(weighted, 0)
      self._moa_loss_op = self._cross_entropy_loss(self.y, moa_logits)
      self._moa_error_op = self._error(self.y, moa_logits)
      moa_grads = tf.gradients(self._moa_loss_op, [self._moa_weights])
      depend_ops = [self._optimizer_dict[i].get_train_op() for i in range(self._n_replicas)]
      
      with tf.control_dependencies(depend_ops):
        self._moa_train_op = tf.assign(self._moa_weights,
                                       self._moa_weights - self._moa_lr*moa_grads[0])


  def flush_summary(self):
    """Flushes accumulated summary to disk."""
    self._summary.flush_summary()

  def create_feed_dict(self, X_batch, y_batch, dataset_type='train'): # pylint: disable=invalid-name
    """Creates feed_dict for session run.

    Args:
      X_batch: input X training batch
      y_batch: input y training batch
      dataset_type: 'train', 'test' or 'validation'

    Returns:
      A dictionary to feed into session's run.
      If `dataset_type` == 'train', adds to feed_dict placeholders
      to store noise (for summary) and sets `is_train` placeholder
      to `True`.
      If `dataset_type` == 'validation'/'test', then doesn't add
      this noise placeholder (since we don't add noise for test or
      validation) and sets `is_train` placeholder to `False`.
      If noise_type is 'dropout' and dataset_type is 'train',
      adds values for keeping parameters during optimization
      (placeholders `keep_prob` for each replica).

    Raises:
      `InvalidDatasetTypeError`: if incorrect `dataset_type`.
    """

    feed_dict = {self.X:X_batch, self.y:y_batch}
    
    if dataset_type in ['test', 'validation'] and 'dropout' in self._noise_type:
      dict_ = {self._noise_plcholders[i]:1.0
               for i in range(self._n_replicas)}

    elif dataset_type in ['test', 'validation'] and 'input_noise' == self._noise_type:
      dict_ = {self._noise_plcholders[i]:0.0
               for i in range(self._n_replicas)}
    
    elif ((dataset_type in ['train'] and 'dropout' in self._noise_type)
          or self._noise_type in ['l2_regularizer', 'input_noise', 'learning_rate']):
      dict_ = {self._noise_plcholders[i]:self._curr_noise_dict[i]
               for i in range(self._n_replicas)}

    elif dataset_type not in ['train', 'test', 'validation']:
      raise InvalidDatasetTypeError()
    
    else:
      dict_ = {}

    feed_dict.update(dict_)
    if dataset_type == 'train':
      feed_dict.update({self.is_train:True})
    else:
      feed_dict.update({self.is_train:False})
    return feed_dict

  def get_ops(self,
              loss_ops=False,
              error_ops=False,
              weights_update_ops=False,
              diffusion_ops=False,
              moa_loss_ops=False,
              moa_error_ops=False,
              moa_weights_update_ops=False,
              moa_weights_vars=False):
    """Returns ops for evaluation in the session context.

    Raises: 
      ValueError: If all arguments are `False` or if one of the MOA
        argument is `True` but mode hasn't been set to `MOA` when
        this class was instantiated.
    """
    res = []
    if loss_ops:
      res += [self._loss_dict[i] for i in range(self._n_replicas)]
    if error_ops:
      res += [self._error_dict[i] for i in range(self._n_replicas)]
    if diffusion_ops:
      res += [self._summary.get_diffusion_ops()]
    if weights_update_ops:
      res += [self._optimizer_dict[i].get_train_op()
              for i in range(self._n_replicas)]
    try:
      if moa_loss_ops:
        res += [self._moa_loss_op]
      if moa_error_ops:
        res += [self._moa_error_op]
      if moa_weights_update_ops:
        res += [self._moa_train_op]
      if moa_weights_vars:
        res += [self._moa_weights]
    except AttributeError:
      err_msg = ('mode argument must be set to `MOA` in order to '
                 'use moa ops.')
      raise ValueError(err_msg)
    
    if len(res) == 0:
      err_msg = ('At least one of the boolean arguments must be '
                 'set to True.')
      raise ValueError(err_msg)
    return res


  def get_train_ops(self, dataset_type='train', loss_err=True, special_ops=False):
    """Returns train ops for session's run.

    The returned list should be used as:
    # evaluated = sess.run(get_train_ops(), feed_dict=...)

    Args:
      dataset_type: One of 'train'/'test'/'validation'
      special_ops: If `True`, adds diffusion.

    Returns:
      A list of train ops.

    Raises:
      `InvalidDatasetTypeError`: if incorrect `dataset_type`.
    """

    if dataset_type not in ['train', 'test', 'validation']:
      raise InvalidDatasetTypeError()
    if loss_err:
      loss = [self._loss_dict[i]
              for i in range(self._n_replicas)]
      error = [self._error_dict[i]
                       for i in range(self._n_replicas)]
      res = loss + error
    else:
      res = []
    

    if special_ops:
      res += self._summary.get_diffusion_ops()

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
      `evaluated`: A list returned by `sess.run(get_train_ops())`
      `step`: A value that is incremented after each mini-batch.
      `epoch`: (Optional) A current epoch that allows accuratly show
        the progress during training via `SummaryExtractor` class.
      `dataset_type`: One of 'train'/'test'/'validation'.

    """

    def has_diffusion_ops():
      """Returns True if evaluated contains diffusion ops."""
      return ((dataset_type in ['test', 'validation']
              and len(evaluated) == 3*self._n_replicas)
            or (dataset_type in ['train']
              and len(evaluated) == 4*self._n_replicas))

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

    # add diffusion, grad norms and weight norms if in the `evaluated`.
    if has_diffusion_ops():
      diffusion = self.extract_evaluated_tensors(evaluated, 'special_vals') 
      for i in range(self._n_replicas):
        self._summary._diffusion_vals[i].append(diffusion[i])

    if epoch is not None:
      self._summary._epoch = epoch

  def add_summary_v2(self, evaluated_dict, epoch, add_noise_vals=False):
    """Logs the summary.

    Args:
      evaluated_dict: A dictionary with keys from 
        `SimulatorGraph._summary_types` and values representing the
        evaluated values of tensors that need to be logged. Each such
        value can be either a list of length `n_replicas` OR a single
        value (in case of batch step) of `summary_type`.
        In case of a list, list must be sorted based on the 
        replica id key.
      epoch: A number of the current epoch.
      add_noise_vals: If `True`, adds noise values of current replica to
        the summary. Should be set to `True` logging train values.

    Raises:
      ValueError: If `summary_type` key in `evaluated_dict` is not
        in `SimulatorGraph._summary_types` or incorrect argument.
    """
    for summary_type in evaluated_dict:
      if summary_type not in self._summary_types:
        err_msg = ('The `summary_type` value is not in list of '
                   'allowed values. Allowed values are:\n'
                   + ', '.join(self._summary_types))
        raise ValueError(err_msg)
      evaluated = evaluated_dict[summary_type]

      if 'loss' in summary_type or 'error' in summary_type or 'steps' in summary_type:
        attr_name = '_' + '_'.join(summary_type.split('_')[:-1])
        attr_name = attr_name.replace('error', 'err')
        attr_name = attr_name.replace('validation', 'valid')
        summary = self._summary.__dict__[attr_name]
      elif 'diffusion' in summary_type:
        attr_name = '_diffusion_vals'
        summary = self._summary.__dict__[attr_name]
      elif 'weights' in summary_type:
        attr_name = '_moa_weights'
        summary = self._summary.__dict__[attr_name]

      try:
        if (isinstance(summary, list)
            and not isinstance(evaluated, list)):
          summary.append(evaluated)
        elif (isinstance(summary, dict)
              and isinstance(evaluated, list)):
          for i in range(self._n_replicas):
            summary[i].append(evaluated[i])
        else:
          print(summary_type, evaluated_dict[summary_type])
          raise ValueError()

      except Exception as exc:

        err_msg = ('`evaluated_dict` must be a dictionary with allowed '
                     'keys (see `SimulatorGraph._summary_types`) with '
                     'values that are floats/integers or lists. Got:\n')
        err_msg += str(evaluated_dict)
        raise ValueError(err_msg) from exc
    if add_noise_vals:
      self._summary.add_noise_vals(self._curr_noise_dict)

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
      `evaluated`: A list returned by sess.run(get_train_ops())
      `tensor_type`: One of 'loss'/'error'/'special_ops'

    Returns:
      A list of specified (by `tensor_type`) tensor values.

    Raises:
      `InvlaidLossFuncError`: Incorrect `tensor_type` value.
      """

    if tensor_type == 'loss': # pylint: disable=no-else-return
      return evaluated[:self._n_replicas]

    elif tensor_type == 'error':
      return evaluated[self._n_replicas:self._n_replicas*2]

    elif tensor_type == 'special_vals': # includes diffusion, grad_norms, weight_norms
      return evaluated[self._n_replicas*2:self._n_replicas*3]

    else:
      raise InvalidTensorTypeError()

  def swap_replicas(self, loss_list):
    """Swaps between two replicas with adjacent noise levels.
    
    Swaps according to:
      1. Uniformly randomly select a pair of adjacent temperatures
        1/beta_i and 1/beta_i+1, for which swap move is proposed.
      2. Accepts swap with probability:
          `min{1, exp(proba_coeff*(beta_i-beta_i+1)*(loss_i-loss_i+1)}`
      3. Updates the acceptance ratio for the proposed swap.

    Args:
      loss_list: A list of evaluated losses for each replica. Should
        be sorted based on replica ID key.
    """
    self._swap_attempts += 1
    random_pair = random.choice(range(self._n_replicas - 1))

    beta = [self._curr_noise_dict[x] for x in range(self._n_replicas)]
    betas_and_ids = [(1.0/b, i) for i, b in enumerate(beta)]
    betas_and_ids.sort(key=lambda x: x[0])
    i = betas_and_ids[random_pair][1]
    j = betas_and_ids[random_pair + 1][1]
    beta_i = betas_and_ids[random_pair][0]
    beta_j = betas_and_ids[random_pair + 1][0]
    li, lj = loss_list[i], loss_list[j]
    proba = np.exp(self._proba_coeff*(li - lj)*(beta_i - beta_j))

    if np.random.uniform() < proba:
      self._curr_noise_dict[i] = beta_j
      self._curr_noise_dict[j] = beta_i
      self._optimizer_dict[i].set_train_route(j)
      self._optimizer_dict[j].set_train_route(i)
      self._swap_successes += 1
      accept_pair = [(i, 1), (j, 1)]
    else:
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

  def _cross_entropy_loss(self, y, logits, clip_value_max=2000.0,
                          regularizer=None, lambda_=None,
                          keep_prob=None): # pylint: disable=invalid-name, no-self-use
    """Cross entropy (possibly) with regularization.
    
    Args:
      y: A placeholder for labels.
      logits: A logits layer of the model.
      clip_by_value: A maximum value of that is possible for loss.
      regularizer: If 'l2_regularizer' a L2-norm regularization
        multiplied by coefficient `lambda_`. If `None`, doesn't add
        regularization to the loss function.
      lambda_: `Placeholder` or `float` - regularization parameter.

    Returns:
      A `Tensor` cross entropy loss.
    """
    with tf.name_scope('cross_entropy'):
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=y, logits=logits)
      loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
      

      if regularizer == 'l2_regularizer_with_dropout':
        raise NotImplementedError()
        if keep_prob is None:
          raise ValueError("For {} the keep_prob must be provided".format(regularizer))

        vars_ = _get_dependencies(loss)
        flatten_vars = [tf.reshape(v, [-1]) for v in vars_]
        concat_vars = tf.concat(flatten_vars, axis=0)
        vars_with_dropout = tf.nn.droput(concat_vars, keep_prob=keep_prob)
        loss = loss + lambda_*tf.nn.l2_loss(vars_with_dropout)

      elif regularizer == 'l2_regularizer':
        vars_ = _get_dependencies(loss)
        l2_vars = [tf.nn.l2_loss(v) for v in vars_]
        loss = loss + lambda_*tf.reduce_sum(l2_vars)

      if clip_value_max is not None:
        loss = tf.clip_by_value(loss, 0.0, clip_value_max)
    return loss

  def _error(self, y, logits): # pylint: disable=invalid-name, no-self-use
    """0-1 error."""
    with tf.name_scope('error'):
        # input arg `predictions` to tf.nn.in_top_k must be of type tf.float32
      
      y_pred = tf.nn.in_top_k(predictions=tf.cast(logits, tf.float32),
                              targets=y,
                              k=1)
      error = 1.0 - tf.reduce_mean(tf.cast(x=y_pred, dtype=DTYPE),
                                   name='error')

    return error

  def _stun_loss(self, loss, gamma=1): # pylint: disable=no-self-use
    """Stochastic tunnelling loss."""
    with tf.name_scope('stun'):
      
      stun = 1 - tf.exp(-gamma*loss) # pylint: disable=no-member

    return stun


  