"""A class that performs simulations."""
import sys
import os
import gc
import json
from time import time
import math

import tensorflow as tf
import sklearn
import numpy as np

from simulator.graph.simulator_graph import SimulatorGraph
from simulator import simulator_utils as s_utils
from simulator.exceptions import IllegalArgumentError

class Simulator: # pylint: disable=too-many-instance-attributes
  """Performs single/multiple simulation for calculating averages.

  This class defines the API for performing simulations. This class
  trains models (possibly multiple times), while class SimulatorGraph
  creates dataflow graphs with duplicated replicas. More functions
  can be added to train models in different setups.

  ### Usage

  ```python
  from tensorflow.examples.tutorials.mnist import input_data
  import numpy as np

  from simulator.simulator import Simulator
  from simulator.summary_extractor import SummaryExtractor
  import simulator.simulator_utils as s_utils
  from simulator.models.mnist_models import nn_mnist_model_dropout
  MNIST_DATAPATH = 'simulator/data/mnist/'

  mnist = input_data.read_data_sets(MNIST_DATAPATH)
  train_data = mnist.train.images
  train_labels = mnist.train.labels
  test_data = mnist.test.images
  test_labels = mnist.test.labels
  valid_data = mnist.validation.images
  valid_labels = mnist.validation.labels

  n_replicas = 8
  separation_ratio = 1.21

  # set simulation parameters
  model_func = nn_mnist_model_dropout
  learning_rate = 0.01
  noise_list = [1/separation_ratio**i for i in range(n_replicas)]
  noise_type = 'dropout_rmsprop'
  batch_size = 200
  n_epochs = 50
  name = 'test_simulation' # simulation name
  test_step = 300 # 1 step==batch_size
  swap_step = 300
  burn_in_period = 400
  loss_func_name = 'cross_entropy'
  description = 'RMSProp with dropout.'
  proba_coeff = 250
  rmsprop_decay = 0.9
  rmsprop_momentum = 0.001
  rmsprop_epsilon=1e-6

  # create and run simulation

  sim = Simulator(
    model=model_func,
    learning_rate=learning_rate,
    noise_list=noise_list,
    noise_type='dropout_rmsprop',
    batch_size=batch_size,
    n_epochs=n_epochs,
    test_step=test_step,
    name=name,
    swap_step=swap_step,
    burn_in_period=burn_in_period,
    loss_func_name='cross_entropy',
    description=description,
    proba_coeff=proba_coeff,
    rmsprop_decay=rmsprop_decay,
    rmsprop_epsilon=rmsprop_epsilon,
    rmsprop_momentum=rmsprop_momentum
    )

  sim.train(train_data=train_data, train_labels=train_labels,
    test_data=test_data, test_labels=test_labels,
    validation_data=valid_data, validation_labels=valid_labels)


  # plot results (possible during training on linux)
  se = SummaryExtractor(name)
  se.show_report()
  ```
  """

  def __init__(self, # pylint: disable=too-many-arguments, too-many-locals
               model,
               learning_rate,
               noise_list,
               noise_type,
               batch_size,
               n_epochs,
               name,
               burn_in_period,
               swap_step,
               separation_ratio,
               ensembles=None,
               n_simulations=1,
               test_step=500,
               tuning_parameter_name=None,
               loss_func_name='cross_entropy',
               verbose_loss='error',
               proba_coeff=1.0,
               description=None,
               test_batch = None,
               hessian=False,
               mode=None,
               moa_lr=None,
               rmsprop_decay=0.9,
               rmsprop_momentum=0.001,
               rmsprop_epsilon=1e-6,
               flush_every=90):
    """Creates a new simulator object.

    Args:
      model: A function that creates inference model (e.g.
        see `simulation.models.nn_mnist_model()`)
      learning_rate: Learning rate for optimizer
      noise_list: A list (not np.array!) for noise/temperatures/dropout
        values. In case of dropout (dropout_rmsprop, dropout_gd), noise_list
        represents the values of KEEPING the neurons, and NOT the probability
        of excluding the neurons.
      noise_type: A string specifying the noise type and optimizer to apply.
        Possible values could be seen at
        `simulation.simulation_builder.graph_builder.SimulatorGraph.__noise_types`
      batch_size: Batch Size
      n_epochs: Number of epochs for each simulation
      name: The name of the simulation. Specifies the a folder name
        from where a summary files can be later accessed.
      n_simulatins: Number of simulation to run.
      test_step: An integer specifing an interval of steps to perform until
        running a test dataset (1 step equals batch_size)
      swap_step: An integer specifying an interval to perform until
        attempting to swap between replicas based on validation dataset.
      separation_ratio: A separation ratio between two adjacent temperatures.
        This value is not important for simulation because the
        noise_list already contains the separated values. This value is
        (as well as some others) are stored in the simulation
        description file (this file is created by _log_params()
        function).
      tuning_parameter_name: As the separation_ratio value, this argument is
        also not important for simulation. It is stored in the description
        file as well.
      burn_in_period: A number of steps until the swaps start to be
        proposed.
      loss_func_name: A function which we want to optimize. Currently,
        only cross_entropy and STUN (stochastic tunneling) are
        supported.
      verbose_loss: A loss to print during training. Default is 0-1 loss.
        Possible parameters are: 'loss', 'error'.
      proba_coeff: The coeffecient is used in calculation of probability
        of swaps. Specifically, we have
        P(accept_swap) = exp(proba_coeff*(beta_1-beta_2)(E_1-E_2))
      description: A custom string that is stored in the description file.
      `test_batch`: A size of a data that is fed during evaluation of loss,
        error etc. for test/validation dataset. For MNIST or similar sized
        dataset the whole test/validation data can be fed at once.
      hessian: (Boolean) If `True`, computes Hessian and its
        eigenvalues during each swap step. Default is `False` since
        it is computationally intensive and should be used only for
        small networks.
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
      rmsprop_decay: Used in
        `simulation.simulation_builder.optimizers.RMSPropOptimizer`
        for noise type `dropout_rmsprop`. This value is ignored for
        other `noise_types`.
      rmsprop_momentum: Used in
        simulation.simulation_builder.optimizers.RMSPropOptimizer
        for noise type 'dropout_rmsprop'. This value is ignored for
        other noise_types.
      rmsprop_epsilon: Used in
        `simulation.simulation_builder.optimizers.RMSPropOptimizer`
        for noise type `dropout_rmsprop`. This value is ignored for
        other `noise_types`.
      flush_every: An integer that defines an interval in seconds that
        currently accumulated training log will be flushed to disk.
        Default is 60 seconds.
    """

    self._model = model
    self._learning_rate = learning_rate
    self._noise_type = noise_type
    self._noise_list = noise_list
    self._n_replicas = len(noise_list)
    self._learning_rate = learning_rate
    self._name = name
    self._n_simulations = n_simulations
    self._burn_in_period = burn_in_period
    self._loss_func_name = loss_func_name
    self._verbose_loss = verbose_loss
    self._proba_coeff = proba_coeff
    self._batch_size = batch_size
    self._n_epochs = n_epochs
    self._test_step = test_step
    self._swap_step = swap_step
    self._separation_ratio = separation_ratio
    self._tuning_param_name = tuning_parameter_name
    self._description = description
    self._ensembles = ensembles
    self._hessian = hessian
    self._mode = mode
    self._moa_lr = moa_lr
    self.rmsprop_decay = rmsprop_decay
    self.rmsprop_momentum = rmsprop_momentum
    self.rmsprop_epsilon = rmsprop_epsilon
    self._test_batch = max(500, batch_size)
    self._logged = False # if log had been written to disk 
    self._flush_every = flush_every
    self._train_step = min(self._test_step, self._swap_step)

  def train_n_times(self, train_data_size=None, **kwargs):
    """Trains `n_simulations` times using the same setup.

    Args:
      `train_data_size`: (Optional) Sets the amount of train data out of
        whole data that is used in simulation. If None, the whole data
        is used. Otherwise, each simulation receives `train_data_size`
        shuffled training data. This value is used when it is needed to
        feed only part of the data to the algorithm.
      `kwargs`: Should be following keyword arguments: `train_data`,
        `train_labels`, `test_data`, `test_labels`, `validation_data`,
        `validation_labels`.
    """

    test_data = kwargs.get('test_data', None)
    test_labels = kwargs.get('test_labels', None)
    valid_data = kwargs.get('validation_data', None)
    valid_labels = kwargs.get('validation_labels', None)
    train_data = kwargs.get('train_data', None)
    train_labels = kwargs.get('train_labels', None)
    sim_names = []

    for i in range(self._n_simulations):
      
      
      train_data, train_labels = sklearn.utils.shuffle(
          train_data, train_labels)

      self._graph = SimulatorGraph(self._model,
                                  self._learning_rate,
                                  self._noise_list,
                                  self._name,
                                  ensembles=self._ensembles,
                                  noise_type=self._noise_type,
                                  simulation_num=i,
                                  loss_func_name=self._loss_func_name,
                                  proba_coeff=self._proba_coeff,
                                  hessian=self._hessian,
                                  mode=self._mode,
                                  moa_lr=self._moa_lr,
                                  rmsprop_decay=self.rmsprop_decay,
                                  rmsprop_momentum=self.rmsprop_momentum,
                                  rmsprop_epsilon=self.rmsprop_epsilon)

      if train_data_size is not None:
        train_data_ = train_data[:train_data_size]
        train_labels_ = train_labels[:train_data_size]
      else:
        train_data_ = train_data
        train_labels_ = train_labels

      self._parallel_tempering_train(train_data=train_data_,
                                     train_labels=train_labels_,
                                     test_data=test_data,
                                     test_labels=test_labels,
                                     validation_data=valid_data,
                                     validation_labels=valid_labels)
      del self._graph
      gc.collect()


  def _parallel_tempering_train(self, **kwargs): # pylint: disable=too-many-locals, invalid-name
    """Trains and swaps between replicas while storing summaries.
    
    Args:
      `kwargs`: Should be following keyword arguments: `train_data`,
        `train_labels`, `test_data`, `test_labels`, `validation_data`,
        `validation_labels`.

    """

    last_flush_time = time()
    
    if not self._logged:
      self._log_params()
      self._logged = True
    try:
      g = self._graph # pylint: disable=invalid-name
    except AttributeError as err:
      if not err.args:
        err.args = ('',)

      err.args = (err.args
                  + ("The SimulatorGraph object is not initialized.",))
      raise

    try:
      train_data = kwargs.get('train_data', None)
      train_labels = kwargs.get('train_labels', None)
      test_data = kwargs.get('test_data', None)
      test_labels = kwargs.get('test_labels', None)
      valid_data = kwargs.get('validation_data', None)
      valid_labels = kwargs.get('validation_labels', None)
      if (train_data is None
          or train_labels is None
          or test_data is None
          or test_labels is None
          or valid_data is None
          or valid_labels is None):
        raise IllegalArgumentError(
            'One of the arguments is None:',
            [x for x in kwargs.keys() if kwargs[x] is None])

      # iterators for train/test/validation
      with g.get_tf_graph().as_default(): # pylint: disable=not-context-manager
        data = tf.data.Dataset.from_tensor_slices({
            'x':train_data,
            'y':train_labels
            }).batch(self._batch_size)
        iterator = data.make_initializable_iterator()
        
        data = tf.data.Dataset.from_tensor_slices({
            'x':valid_data,
            'y':valid_labels
            }).batch(self._test_batch)
        iter_valid = data.make_initializable_iterator()

        data = tf.data.Dataset.from_tensor_slices({
            'x':test_data,
            'y':test_labels
            }).batch(self._test_batch)
        iter_test = data.make_initializable_iterator()

    except: # pylint: disable=try-except-raise
      raise

    step = 1

    with tf.Session(graph=g.get_tf_graph()) as sess:
      _ = sess.run([iterator.initializer,
                    iter_valid.initializer,
                    iter_test.initializer,
                    g.variable_initializer])

      next_batch = iterator.get_next()
      next_batch_test = iter_test.get_next()
      next_batch_valid = iter_valid.get_next()

      train_batch_loss = {i:[] for i in range(self._n_replicas)}
      train_batch_err = {i:[] for i in range(self._n_replicas)}

      for epoch in range(self._n_epochs):
        while True:
          try:
            

            ### test ###
            if step % self._test_step == 0 or step == 1:
              evaluated = self._train_epoch(sess,
                                            next_batch_test,
                                            iter_test,
                                            dataset_type='test')
              g.add_summary_v2({
                'test_loss_summary': evaluated[:self._n_replicas],
                'test_error_summary': evaluated[self._n_replicas:],
                'test_steps_summary': step
                }, epoch)

              self.print_log(epoch,
                             step,
                             evaluated[self._n_replicas:],
                             g.get_accept_ratio())

            ### validation + swaps ###
            if step % self._swap_step == 0:
              evaluated = self._train_epoch(sess,
                                            next_batch_valid,
                                            iter_valid,
                                            dataset_type='validation')
              g.add_summary_v2({
                'validation_loss_summary': evaluated[:self._n_replicas],
                'validation_error_summary': evaluated[self._n_replicas:],
                'validation_steps_summary': step
                }, epoch)
              if step > self._burn_in_period:
                g.swap_replicas(evaluated[:self._n_replicas])

            ### train ###
            batch = sess.run(next_batch)
            step += 1
            # Forward Pass Only to compute error/losses
            feed_dict = g.create_feed_dict(batch['x'], batch['y'], dataset_type='test')
            loss_err_ops = g.get_ops(loss_ops=True, error_ops=True)
            loss_err_vals = sess.run(loss_err_ops, feed_dict=feed_dict)
            for i in range(self._n_replicas):
              train_batch_loss[i].append(loss_err_vals[i])
              train_batch_err[i].append(loss_err_vals[self._n_replicas + i])

            # Compute diffusion and append values to summaries
            if step % self._train_step == 0 or step == 1:
              diff_ops = g.get_ops(diffusion_ops=True)
              evaled_diffs = sess.run(diff_ops)
              g.add_summary_v2({
                'diffusion_summary': evaled_diffs[0],
                'train_loss_summary': [np.mean(train_batch_loss[i]) for i in range(self._n_replicas)],
                'train_error_summary':[np.mean(train_batch_err[i]) for i in range(self._n_replicas)],
                'train_steps_summary': step,
                }, epoch, add_noise_vals=True)

              del train_batch_loss
              del train_batch_err

              train_batch_loss = {i:[] for i in range(self._n_replicas)}
              train_batch_err = {i:[] for i in range(self._n_replicas)}

            # Update weights step
            update_ops = g.get_ops(weights_update_ops=True)
            feed_dict = g.create_feed_dict(batch['x'], batch['y'])
            _ = sess.run(update_ops, feed_dict=feed_dict)

            ############ old impl
            '''
            if step % self._train_step == 0 or step == 1:
              special_ops = True
            else:
              special_ops = False

            batch = sess.run(next_batch)

            #if self._noise_type not in ['dropout', 'dropout_gd', 'dropout_rmsprop']:

            if special_ops:
              ops = g.get_train_ops(special_ops=special_ops, loss_err=False)
              diff_ops = ops[:g._n_replicas]
              evaled_diffs = sess.run(diff_ops)

            feed_dict = g.create_feed_dict(batch['x'], batch['y'])

            loss_err_train_ops = g.get_train_ops()
            loss_err_ops = loss_err_train_ops[:2*self._n_replicas]
            train_ops = loss_err_train_ops[-self._n_replicas:]
            loss_err_vals = sess.run(loss_err_ops, feed_dict=feed_dict)
            train_vals = sess.run(train_ops, feed_dict=feed_dict)

            evaluated = (loss_err_vals
                        + evaled_diffs
                        + train_vals)

            loss = g.extract_evaluated_tensors(evaluated, 'loss')
            err = g.extract_evaluated_tensors(evaluated, 'error')


            for i in range(self._n_replicas):
                train_batch_loss[i].append(loss[i])
                train_batch_err[i].append(err[i])

            if step % self._train_step == 0 or step == 1:
              evaled = [np.mean(train_batch_loss[i]) for i in range(self._n_replicas)]
              evaled += [np.mean(train_batch_err[i]) for i in range(self._n_replicas)]
              evaled += g.extract_evaluated_tensors(evaluated, 'special_vals')

              g.add_summary(evaled+evaluated[-self._n_replicas:],
                            step=step,
                            epoch=epoch,
                            dataset_type='train')

              del train_batch_loss
              del train_batch_err

              train_batch_loss = {i:[] for i in range(self._n_replicas)}
              train_batch_err = {i:[] for i in range(self._n_replicas)}
            '''
            if time() - last_flush_time > self._flush_every:
              g.flush_summary()
              last_flush_time = time()

          except tf.errors.OutOfRangeError:
            sess.run(iterator.initializer)
            break
    '''
    if not train_batch_loss[0]:
      evaled = [np.mean(train_batch_loss[i]) for i in range(self._n_replicas)]
      evaled += [np.mean(train_batch_err[i]) for i in range(self._n_replicas)]
      g.add_summary(evaled, step=step, epoch=self._n_epochs, dataset_type='train')
    '''
    g._summary._latest_epoch = self._n_epochs
    g.flush_summary()

  def _parallel_tempering_train_moa(self, **kwargs): # pylint: disable=too-many-locals, invalid-name
    """Trains and swaps between replicas while storing summaries.
    
    This function differs from others by the fact that it supports MOA.
    Args:
      `kwargs`: Should be following keyword arguments: `train_data`,
        `train_labels`, `test_data`, `test_labels`, `validation_data`,
        `validation_labels`.

    """

    last_flush_time = time()
    
    if not self._logged:
      self._log_params()
      self._logged = True
    try:
      g = self._graph # pylint: disable=invalid-name
    except AttributeError as err:
      if not err.args:
        err.args = ('',)

      err.args = (err.args
                  + ("The SimulatorGraph object is not initialized.",))
      raise

    try:
      train_data = kwargs.get('train_data', None)
      train_labels = kwargs.get('train_labels', None)
      test_data = kwargs.get('test_data', None)
      test_labels = kwargs.get('test_labels', None)
      valid_data = kwargs.get('validation_data', None)
      valid_labels = kwargs.get('validation_labels', None)
      if (train_data is None
          or train_labels is None
          or test_data is None
          or test_labels is None
          or valid_data is None
          or valid_labels is None):
        raise IllegalArgumentError(
            'One of the arguments is None:',
            [x for x in kwargs.keys() if kwargs[x] is None])

      # iterators for train/test/validation
      with g.get_tf_graph().as_default(): # pylint: disable=not-context-manager
        data = tf.data.Dataset.from_tensor_slices({
            'x':train_data,
            'y':train_labels
            }).batch(self._batch_size)
        iterator = data.make_initializable_iterator()
        
        data = tf.data.Dataset.from_tensor_slices({
            'x':valid_data,
            'y':valid_labels
            }).batch(self._test_batch)
        iter_valid = data.make_initializable_iterator()

        data = tf.data.Dataset.from_tensor_slices({
            'x':test_data,
            'y':test_labels
            }).batch(self._test_batch)
        iter_test = data.make_initializable_iterator()

    except: # pylint: disable=try-except-raise
      raise

    step = 1

    with tf.Session(graph=g.get_tf_graph()) as sess:
      _ = sess.run([iterator.initializer,
                    iter_valid.initializer,
                    iter_test.initializer,
                    g.variable_initializer])

      next_batch = iterator.get_next()
      next_batch_test = iter_test.get_next()
      next_batch_valid = iter_valid.get_next()

      train_batch_loss = {i:[] for i in range(self._n_replicas)}
      train_batch_err = {i:[] for i in range(self._n_replicas)}
      moa_batch_loss = []
      moa_batch_err = []

      for epoch in range(self._n_epochs):

        while True:
          
          try:
            

            ### test ###
            if step % self._test_step == 0 or step == 1:
              evaluated = self._train_epoch(sess,
                                            next_batch_test,
                                            iter_test,
                                            dataset_type='test')
              test_losses = evaluated[:self._n_replicas]
              test_errs = evaluated[self._n_replicas:]

              g.add_summary_v2({
                'test_loss_summary': test_losses,
                'test_error_summary': test_errs,
                'test_steps_summary': step
                }, epoch)
              # MOA test metrics
              test_loss, test_err = self._train_epoch(sess,
                                                      next_batch_test,
                                                      iter_test,
                                                      dataset_type='test',
                                                      moa=True)
              g.add_summary_v2({'moa_test_loss_summary': test_loss,
                                'moa_test_error_summary': test_err,
                                'moa_test_steps_summary': step}, epoch)

              if 'loss' in self._verbose_loss:
                verbose_loss_vals = test_losses + [test_loss]
              else:
                verbose_loss_vals = test_errs + [test_err]
              self.print_log(epoch,
                             step,
                             verbose_loss_vals,
                             g.get_accept_ratio())

            ### validation + swaps ###
            if step % self._swap_step == 0 or step == 1:
              evaluated = self._train_epoch(sess,
                                            next_batch_valid,
                                            iter_valid,
                                            dataset_type='validation')
              valid_losses = evaluated[:self._n_replicas]
              valid_errs = evaluated[self._n_replicas:]
              g.add_summary_v2({
                'validation_loss_summary': valid_losses,
                'validation_error_summary': valid_errs,
                'validation_steps_summary': step
                }, epoch)
              if step > self._burn_in_period:
                g.swap_replicas(evaluated[:self._n_replicas])

              valid_loss, valid_err = self._train_epoch(sess,
                                                        next_batch_test,
                                                        iter_test,
                                                        dataset_type='test',
                                                        moa=True)
              g.add_summary_v2({'moa_validation_loss_summary': valid_loss,
                                'moa_validation_error_summary': valid_err,
                                'moa_validation_steps_summary': step}, epoch)

            ### train ###
            batch = sess.run(next_batch)
            step += 1
            rep_train_loss_ops = g.get_ops(loss_ops=True)
            rep_train_err_ops = g.get_ops(error_ops=True)
            moa_train_loss_ops = g.get_ops(moa_loss_ops=True)
            moa_train_err_ops = g.get_ops(moa_error_ops=True)

            execute_ops = (rep_train_loss_ops
                          + rep_train_err_ops
                          + moa_train_loss_ops
                          + moa_train_err_ops)

            feed_dict = g.create_feed_dict(batch['x'], batch['y'], dataset_type='test')

            evaled_ops = sess.run(execute_ops, feed_dict=feed_dict)

            rep_train_loss_vals = evaled_ops[:self._n_replicas]
            rep_train_err_vals = evaled_ops[self._n_replicas:2*self._n_replicas]
            moa_train_loss_val = evaled_ops[2*self._n_replicas]
            moa_train_err_val = evaled_ops[2*self._n_replicas + 1]

            for i in range(self._n_replicas):
              train_batch_loss[i].append(rep_train_loss_vals[i])
              train_batch_err[i].append(rep_train_err_vals[i])

            moa_batch_loss.append(moa_train_loss_val)
            moa_batch_err.append(moa_train_err_val)

            if step % self._train_step == 0 or step == 1:
              diff_ops = g.get_ops(diffusion_ops=True)
              feed_dict = g.create_feed_dict(batch['x'], batch['y'])
              evaled_diffs = sess.run(diff_ops, feed_dict=feed_dict)
              moa_weights_ops = g.get_ops(moa_weights_vars=True)
              moa_weights_vals = sess.run(moa_weights_ops)
              g.add_summary_v2({
                'diffusion_summary': evaled_diffs[0],
                'moa_train_loss_summary': np.mean(moa_batch_loss),
                'moa_train_error_summary': np.mean(moa_batch_err),
                'moa_weights_summary':list(moa_weights_vals[0].squeeze()),
                'moa_train_steps_summary': step,
                'train_loss_summary': [np.mean(train_batch_loss[i]) for i in range(self._n_replicas)],
                'train_error_summary':[np.mean(train_batch_err[i]) for i in range(self._n_replicas)],
                'train_steps_summary': step,
                }, epoch, add_noise_vals=True)

              del train_batch_loss
              del train_batch_err
              del moa_batch_err
              del moa_batch_loss

              train_batch_loss = {i:[] for i in range(self._n_replicas)}
              train_batch_err = {i:[] for i in range(self._n_replicas)}
              moa_batch_loss = []
              moa_batch_err = []

            rep_updates = g.get_ops(weights_update_ops=True)
            moa_updates = g.get_ops(moa_weights_update_ops=True)

            feed_dict = g.create_feed_dict(batch['x'],
                                           batch['y'],
                                           dataset_type='train')
            _ = sess.run(rep_updates + moa_updates, feed_dict=feed_dict)

            if time() - last_flush_time > self._flush_every:
              g.flush_summary()
              last_flush_time = time()

          except tf.errors.OutOfRangeError:
            sess.run(iterator.initializer)
            break
    '''
    if not train_batch_loss[0]:
      evaled = [np.mean(train_batch_loss[i]) for i in range(self._n_replicas)]
      evaled += [np.mean(train_batch_err[i]) for i in range(self._n_replicas)]
      g.add_summary(evaled, step=step, epoch=self._n_epochs, dataset_type='train')
    '''
    g._summary._latest_epoch = self._n_epochs
    g.flush_summary()

  def _parallel_tempering_train_v2(self, **kwargs): # pylint: disable=too-many-locals, invalid-name
    """Trains and swaps between replicas while storing summaries.

    This function differs from `_parallel_tempering_train()` by the
    fact that it calculates loss/error on the whole train data set
    and just accumulates loss/error over batches. This gives
    better understanding of the performance on train data especially
    when we're dealing with dropout that applied only during training
    but not during testing, which may result in better perfomance on
    test data than on train data. 
    
    Args:
      `kwargs`: Should be following keyword arguments: `train_data`,
        `train_labels`, `test_data`, `test_labels`, `validation_data`,
        `validation_labels`.

    """

    last_flush_time = time()
    
    if not self._logged:
      self._log_params()
      self._logged = True
    try:
      g = self._graph # pylint: disable=invalid-name
    except AttributeError as err:
      if not err.args:
        err.args = ('',)

      err.args = (err.args
                  + ("The SimulatorGraph object is not initialized.",))
      raise

    try:
      train_data = kwargs.get('train_data', None)
      train_labels = kwargs.get('train_labels', None)
      test_data = kwargs.get('test_data', None)
      test_labels = kwargs.get('test_labels', None)
      valid_data = kwargs.get('validation_data', None)
      valid_labels = kwargs.get('validation_labels', None)
      if (train_data is None
          or train_labels is None
          or test_data is None
          or test_labels is None
          or valid_data is None
          or valid_labels is None):
        raise IllegalArgumentError(
            'One of the arguments is None:',
            [x for x in kwargs.keys() if kwargs[x] is None])

      # iterators for train/test/validation
      with g.get_tf_graph().as_default(): # pylint: disable=not-context-manager
        data = tf.data.Dataset.from_tensor_slices({
            'x':train_data,
            'y':train_labels
            }).batch(self._batch_size)
        iterator = data.make_initializable_iterator()
        
        data = tf.data.Dataset.from_tensor_slices({
            'x':train_data,
            'y':train_labels
          }).batch(self._batch_size)
        iter_train = data.make_initializable_iterator()

        data = tf.data.Dataset.from_tensor_slices({
            'x':valid_data,
            'y':valid_labels
            }).batch(self._test_batch)
        iter_valid = data.make_initializable_iterator()

        data = tf.data.Dataset.from_tensor_slices({
            'x':test_data,
            'y':test_labels
            }).batch(self._test_batch)
        iter_test = data.make_initializable_iterator()

    except: # pylint: disable=try-except-raise
      raise

    step = 0

    with tf.Session(graph=g.get_tf_graph()) as sess:
      _ = sess.run([iterator.initializer,
                    iter_valid.initializer,
                    iter_test.initializer,
                    iter_train.initializer,
                    g.variable_initializer])

      next_batch = iterator.get_next()
      next_batch_train = iterator.get_next()
      next_batch_test = iter_test.get_next()
      next_batch_valid = iter_valid.get_next()

      for epoch in range(self._n_epochs):

        while True:

          try:
            step += 1

            ### test ###
            if step % self._test_step == 0 or step == 1:
              evaluated = self._train_epoch(sess,
                                            next_batch_test,
                                            iter_test,
                                            dataset_type='test')
              test_losses = evaluated[:self._n_replicas]
              test_errs = evaluated[self._n_replicas:]
              g.add_summary_v2({
                'test_loss_summary': test_losses,
                'test_error_summary': test_errs,
                'test_steps_summary': step
                })
              
              if 'loss' in self._verbose_loss:
                verbose_loss_vals = test_losses
              else:
                verbose_loss_vals = test_errs
              self.print_log(epoch,
                             step,
                             verbose_loss_vals,
                             g.get_accept_ratio())

            ### validation + swaps ###
            if step % self._swap_step == 0 or step == 1:
              evaluated = self._train_epoch(sess,
                                            next_batch_valid,
                                            iter_valid,
                                            dataset_type='validation')
              valid_losses = evaluated[:self._n_replicas]
              valid_errs = evaluated[self._n_replicas:]
              g.add_summary_v2({
                'validation_loss_summary': valid_losses,
                'validation_error_summary': valid_errs,
                'validation_steps_summary': step
                })
              if step > self._burn_in_period:
                g.swap_replicas(evaluated[:self._n_replicas])

            ### train ###
            if step % self._train_step == 0 or step == 1:
              diff_ops = g.get_ops(diffusion_ops=True)
              evaled_diffs = sess.run(diff_ops)
              loss_err_vals = self._train_epoch(sess,
                                                next_batch_train,
                                                iter_train,
                                                dataset_type='train')

              g.add_summary_v2({
                'train_loss_summary': loss_err_vals[:self._n_replicas],
                'train_error_summary': loss_err_vals[self._n_replicas:],
                'train_steps_summary': step,
                'diffusion_summary': evaled_diffs
                })

            batch = sess.run(next_batch)
            feed_dict = g.create_feed_dict(batch['x'], batch['y'])
            update_ops = g.get_ops(weights_update_ops=True)

            _ = sess.run(update_ops, feed_dict=feed_dict)

            if time() - last_flush_time > self._flush_every:
              g.flush_summary()
              last_flush_time = time()

          except tf.errors.OutOfRangeError:
            sess.run(iterator.initializer)
            break

    g._summary._latest_epoch = self._n_epochs
    g.flush_summary()

  def _train_epoch(self, sess, next_batch, iterator, dataset_type, moa=False):
    """Trains the whole data in the iterator and returns average results."""
    batch_loss = {i:[] for i in range(self._n_replicas)}
    batch_err = {i:[] for i in range(self._n_replicas)}

    g = self._graph

    while True:
      try:
        batch = sess.run(next_batch)
        feed_dict = g.create_feed_dict(batch['x'],
                                       batch['y'],
                                       dataset_type=dataset_type)
        if moa is True:
          ops = g.get_ops(moa_loss_ops=True,
                          moa_error_ops=True)
          evaluated = sess.run(ops, feed_dict=feed_dict)
          loss = evaluated[0]
          err = evaluated[1]
          batch_loss[0].append(loss)
          batch_err[0].append(err)
        else:
          ops = g.get_ops(loss_ops=True,
                          error_ops=True)

          evaluated = sess.run(ops, feed_dict=feed_dict)
          loss = evaluated[:self._n_replicas]
          err = evaluated[self._n_replicas:]
          for i in range(self._n_replicas):
            batch_loss[i].append(loss[i])
            batch_err[i].append(err[i])

      except tf.errors.OutOfRangeError:
        if moa is False:
          loss = [np.mean(batch_loss[i]) for i in range(self._n_replicas)]
          err = [np.mean(batch_err[i]) for i in range(self._n_replicas)]
          res = loss + err
        else:
          loss = np.mean(batch_loss[0])
          err = np.mean(batch_err[0])
          res = [loss, err]
        sess.run(iterator.initializer)
        break

    return res


  def train(self, train_data_size=None, **kwargs):
    """Trains model a single time using parallel tempering setup.
    
    Args:
      `train_data_size`: (Optional) The size of the train_data. This
        can be used in case where only part of data should be used
        while training. It is also possible to pass only a fraction
        of the data keeping this argument `None`. If `None`, uses
        the whole data.
      `kwargs`: Should be following keyword arguments: `train_data`,
        `train_labels`, `test_data`, `test_labels`, `validation_data`,
        `validation_labels`.
    """
    self._graph = SimulatorGraph(self._model,
                                self._learning_rate,
                                self._noise_list,
                                self._name,
                                ensembles=self._ensembles,
                                noise_type=self._noise_type,
                                simulation_num=0,
                                loss_func_name=self._loss_func_name,
                                proba_coeff=self._proba_coeff,
                                hessian=self._hessian,
                                mode=self._mode,
                                moa_lr=self._moa_lr,
                                rmsprop_decay=self.rmsprop_decay,
                                rmsprop_momentum=self.rmsprop_momentum,
                                rmsprop_epsilon=self.rmsprop_epsilon)

    

    
    test_data = kwargs.get('test_data', None)
    test_labels = kwargs.get('test_labels', None)
    valid_data = kwargs.get('validation_data', None)
    valid_labels = kwargs.get('validation_labels', None)
    train_data = kwargs.get('train_data', None)
    train_labels = kwargs.get('train_labels', None)
    
    if self._mode in ['MOA', 'mixture_of_experts', 'moa']:
      pt_fn = self._parallel_tempering_train_moa
    else:
      pt_fn = self._parallel_tempering_train
    
    if train_data_size is not None:
      train_data, train_labels = sklearn.utils.shuffle(
          train_data, train_labels)
      train_data_ = train_data[:train_data_size]
      train_labels_ = train_labels[:train_data_size]
    else:
      train_data_ = train_data
      train_labels_ = train_labels
    pt_fn(train_data=train_data_,
          train_labels=train_labels_,
          test_data=test_data,
          test_labels=test_labels,
          validation_data=valid_data,
          validation_labels=valid_labels)
  
  def _initialize_uninitialized(self, sess):
    with self._graph.get_tf_graph().as_default():
      global_vars = tf.global_variables()
      is_not_initialized = sess.run([tf.is_variable_initialized(v)
                                     for v in global_vars])
      not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized)
                              if not f]
      if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

  def _log_params(self):
    """Creates and stores description file."""
    dirpath = self._graph.get_summary_dirname()
    filepath = os.path.join(dirpath, 'description.json')
    if not os.path.exists(dirpath):
      os.makedirs(dirpath)
    _log = {
        'name':self._name,
        'noise_type': self._noise_type,
        'noise_list': self._noise_list,
        'n_replicas': len(self._noise_list),
        'learning_rate':self._learning_rate,
        'n_epochs':self._n_epochs,
        'batch_size':self._batch_size,
        'swap_step': self._swap_step,
        'separation_ratio': self._separation_ratio,
        'n_simulations': self._n_simulations,
        'tuning_parameter_name':self._tuning_param_name,
        'description':self._description,
        'burn_in_period':self._burn_in_period,
        'proba_coeff':self._proba_coeff,
        'n_params': int(self._graph._n_params),
        'mode': self._mode,
        'moa_lr': self._moa_lr
    }
    with open(filepath, 'w') as file:
      json.dump(_log, file, indent=4)

  def print_log(self, # pylint: disable=too-many-arguments
                epoch,
                step,
                loss,
                accept_ratio):
    """Helper for logs during training."""
    names = ['epoch:'+str(epoch),
             'step:'+str(step),
             'loss:'+str(loss),
             'accept_proba:'+str(accept_ratio)]
    buff = ', '.join(names) + '       '

    self.stdout_write(buff)

  def stdout_write(self, buff): # pylint: disable=no-self-use
    """Writes to stdout buffer with beginning of the line character."""
    sys.stdout.write('\r' + buff)
    sys.stdout.flush()
