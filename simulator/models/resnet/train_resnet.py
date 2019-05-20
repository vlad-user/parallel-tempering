import sys
import os
import json
from time import time
import pickle

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as arys_shuffle

from simulator.models.resnet import resnet_v2
from simulator.graph.device_placer import _get_available_gpus
from simulator.graph.summary import Summary
from simulator import simulator_utils as s_utils
from simulator.read_datasets import _create_cifar_data_or_get_existing_resnet

_FLUSH_EVERY = 90
_MAX_STEPS = 64000

def _diffusion_ops():
  curr_vars = tf.trainable_variables()
  init_vars = [tf.Variable(v.initialized_value(), trainable=False) for v in curr_vars]
  difference = [curr_val - init_val
                for curr_val, init_val in zip(curr_vars, init_vars)]
  l2_difference = [2*tf.nn.l2_loss(d) for d in difference]
  diffusion = tf.reduce_sum(l2_difference)
  return diffusion

def create_resnet_model(resnet_size=20, input_shape=(32, 32, 3), momentum=0.9,):
  with tf.device('gpu:1'):
    with tf.name_scope('inputs'):
      x = tf.placeholder(tf.float32, shape=(None, ) + input_shape)
      y = tf.placeholder(tf.int32, shape=(None))

    #logits = resnet_creator(x, resnet_size)
    N = int((resnet_size - 2) / 6)
    logits, istrain = resnet_v2.resnet_creator(x, N)

    with tf.name_scope('learning_rate'):
      lr = tf.placeholder(tf.float32, shape=())
    
    with tf.name_scope('loss'):
      xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=y, logits=logits)
      loss_tensor = tf.reduce_mean(xentropy)

    with tf.name_scope('error'):
      y_pred = tf.nn.in_top_k(predictions=tf.cast(logits, tf.float32),
                              targets=y,
                              k=1)
      error_tensor = 1.0 - tf.reduce_mean(tf.cast(y_pred, tf.float32))

    with tf.name_scope('diffusion'):
      diffusion_tensor = _diffusion_ops()

    with tf.name_scope('optimizer'):
      optimizer = tf.train.MomentumOptimizer(lr, momentum=momentum)
      train_op = optimizer.minimize(loss_tensor)


    return x, y, lr, istrain, loss_tensor, error_tensor, diffusion_tensor, train_op

def _train_epoch(sess, next_batch, iterator, loss_tensor, err_tensor, inputs):
  """Trains the whole data in the iterator and returns average results."""
  batch_loss = []
  batch_err = []
  x, y = inputs

  while True:
    try:
      batch = sess.run(next_batch)
      feed_dict = {x: batch['x'],
                   y: batch['y'],
                   }
      
      loss_val, err_val = sess.run([loss_tensor, err_tensor],
                                   feed_dict={x: batch['x'], y: batch['y']})
      batch_loss.append(loss_val)
      batch_err.append(err_val)

    except tf.errors.OutOfRangeError:
      sess.run(iterator.initializer)
      break

  return np.mean(batch_loss), np.mean(batch_err)

def train(n_epochs, steps, batch_size=128, simulation_num=0, name=None):
  train_step = steps['train_step']
  test_step = steps['test_step']
  valid_step = steps['valid_step']
  
  x_train, y_train, x_test, y_test, x_valid, y_valid = (
      _create_cifar_data_or_get_existing_resnet())

  if name == None:
    name = s_utils.generate_experiment_name(
        model_name='resnet20',
        dataset_name='cifar',
        separation_ratio=0,
        n_replicas=1,
        noise_type='momentumnonoise',
        n_epochs=n_epochs,
        burn_in_period=0,
        beta_0=0,
        beta_n=0,
        loss_func_name='crossentropy',
        swap_step=valid_step,
        learning_rate=0.0,
        batch_size=batch_size,
        proba_coeff=0,
        train_data_size=45000,
        mode=None)

  print(name)



  last_flush_time = time()
  
  x, y, lr, istrain, loss_tensor, error_tensor, diffusion_tensor, train_op = (
      create_resnet_model())

  diffusion_ops = {0:diffusion_tensor}
  summary = Summary(name,
                    {0:[]},
                    {0:[]},
                    simulation_num,
                    diffusion_ops=diffusion_ops)

  # log params
  dirpath = summary.get_dirname()
  filepath = os.path.join(dirpath, 'description.json')
  log = {
    'name': name,
    'noise_type':'momentnonoise',
    'noise_list':[0.1, 0.01, 0.001],
    'n_replicas': 1,
    'learning_rate': 0.1,
    'n_epochs': n_epochs,
    'batch_size': batch_size,
    'swap_step': valid_step,
    'separation_ratio': 0,
    'n_simulations': 1,
    'tuning_parameter_name': None,
    'description': 'simulation',
    'burn_in_period': 0,
    'proba_coeff': 0,
    'n_params': int(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])),
    'mode': None,
    'moa_lr': None
  }
  with open(filepath, 'w') as fo:
    json.dump(log, fo, indent=4)
  
  
  

  '''
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  data, labels = np.vstack([x_train, x_test]), np.vstack([y_train, y_test])
  data = data / 255.
  labels = labels.squeeze()
  mean = np.mean(data, axis=0)[None, ...]
  data = data - mean
  data, labels = arys_shuffle(data, labels)
  x_test, y_test = data[:10000], labels[:10000]
  x_train, y_train = data[10000:], labels[10000:]
  x_train, x_valid, y_train, y_valid = train_test_split(x_train,
                                                      y_train,
                                                      test_size=0.1,
                                                      shuffle=True)

  '''

  data = tf.data.Dataset.from_tensor_slices({
      'x': x_train,
      'y': y_train
    }).shuffle(x_train.shape[0]).batch(batch_size)
  iter_train = data.make_initializable_iterator()

  data = tf.data.Dataset.from_tensor_slices({
      'x': x_test,
      'y': y_test
    }).batch(batch_size)
  iter_test = data.make_initializable_iterator()

  data = tf.data.Dataset.from_tensor_slices({
      'x': x_valid,
      'y': y_valid
    }).batch(batch_size)
  iter_valid = data.make_initializable_iterator()

  config = tf.ConfigProto(allow_soft_placement=True)
  with tf.Session(config=config) as sess:
    sess.run([tf.global_variables_initializer(),
              iter_train.initializer,
              iter_test.initializer,
              iter_valid.initializer])

    next_batch_train = iter_train.get_next()
    next_batch_test = iter_test.get_next()
    next_batch_valid = iter_valid.get_next()

    batch_loss = []
    batch_err = []
    step = 1

    for epoch in range(n_epochs):
      while True:
        try:
          

          # test
          if step % test_step == 0 or step == 1 or step == _MAX_STEPS:
            loss_val, err_val = _train_epoch(sess,
                                             next_batch_test,
                                             iter_test,
                                             loss_tensor,
                                             error_tensor,
                                             (x, y))
            summary._test_loss[0].append(loss_val)
            summary._test_err[0].append(err_val)
            summary._test_steps.append(step)
            loss_to_log = loss_val
            err_to_log = err_val

          log = {'epoch': epoch + 1,
                 'step': step,
                 'loss': loss_to_log,
                 'error': err_to_log}
          print_log(log)

          # validation
          if step % valid_step == 0 or step == 1 or step == _MAX_STEPS:
            loss_val, err_val = _train_epoch(sess,
                                             next_batch_valid,
                                             iter_valid,
                                             loss_tensor,
                                             error_tensor,
                                             (x, y))
            summary._valid_loss[0].append(loss_val)
            summary._valid_err[0].append(err_val)
            summary._valid_steps.append(step)

          # train
          batch = sess.run(next_batch_train)
          step += 1
          if step <= 32000:
            lr_val = 0.1
          elif 32000 < step <= 48000:
            lr_val = 0.01
          else:
            lr_val = 0.001

          feed_dict = {
              x: batch['x'],
              y: batch['y'],
              lr: lr_val,
              istrain: True
          }
          loss_val, err_val, _ = sess.run([loss_tensor, error_tensor, train_op],
                                          feed_dict=feed_dict)
          batch_loss.append(loss_val)
          batch_err.append(err_val)

          if step % train_step == 0 or step == 1 or step == _MAX_STEPS:
            diffusion_ops = summary.get_diffusion_ops()
            diffusion_vals = sess.run(diffusion_ops)
            summary._train_noise_vals[0].append(lr_val)
            summary._diffusion_vals[0].append(diffusion_vals[0])
            summary._train_loss[0].append(np.mean(batch_loss))
            summary._train_err[0].append(np.mean(batch_err))
            summary._train_steps.append(step)
            summary._epoch = epoch
            batch_loss = []
            batch_err = []

          if time() - last_flush_time > _FLUSH_EVERY:
            summary.flush_summary()
            last_flush_time = time()

          if step > _MAX_STEPS:
            summary.flush_summary()
            print()
            print('DONE')
            sys.exit()
          
        except tf.errors.OutOfRangeError:
          sess.run(iter_train.initializer)
          break


def print_log(dict_):

  buff = '\r[epoch:{0}]|[step:{1}/{2}]|[loss:{3:.4f}]|[err:{4:.4f}]'.format(
      dict_['epoch'], dict_['step'], _MAX_STEPS, dict_['loss'], dict_['error'])
  sys.stdout.write(buff)
  sys.stdout.flush()