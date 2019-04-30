"""This module is not a part of the package. It is for the purpose of
verifying some results using pytorch."""
import os
import random
import json
import sys
import pickle

import numpy as np
import torch
from torch.utils.data import dataset

from simulator.testing import models
from simulator.read_datasets import get_cifar10_data
from simulator.read_datasets import get_cifar10_data_debug

FNAME_PREFIX = 'summary_'
FNAME_SUFFIX = '.log'


DEVICE = 0
def compute_accuracy(y_pred, y):
   if list(y_pred.size()) != list(y.size()):
      raise ValueError('Inputs have different shapes.',
                       list(y_pred.size()), 'and', list(y.size()))

  result = [1 if y1==y2 else 0 for y1, y2 in zip(y_pred, y)]

  return sum(result) / len(result)

def compute_zero_one_error(y_pred, y):
  return 1.0 - compute_accuracy(y_pred.to('cpu'), y.type(torch.LongTensor).to('cpu'))


def train_cifar(Model,
                name='lenet5_torch_test',
                batch_size=50,
                n_epochs=2000,
                learning_rate=0.01,
                swap_step=300,
                test_step=500,
                proba_coeff=50,
                noise_list=None,
                burn_in_period=2000,
                train_data_size=5000,
                debug=False,
                replace_existing=False):
  ######################################################
  def run_test_epoch(model, pytorch_loss, loader):
    errs, losses = [], []
    for (x, y) in loader:
      x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
      x, y = x.to(DEVICE), y.to(DEVICE)
      y_pred = model(x)
      loss_val = pytorch_loss(y_pred.type(torch.FloatTensor), y.type(torch.LongTensor))
      vals, y_pred = torch.max(y_pred, 1)
      error = compute_zero_one_error(y_pred, y)
      losses.append(loss_val.item())
      errs.append(error)

    loss = np.mean(losses)
    error = np.mean(errs)

    return loss, error

  ######################################################

  ######################################################
  if debug:
    x_train, y_train, x_test, y_test, x_valid, y_valid = get_cifar10_data_debug(
        train_data_size=train_data_size, replace_existing=replace_existing)
  else:
    x_train, y_train, x_test, y_test, x_valid, y_valid = get_cifar10_data()

  x_train, x_test, x_valid = (np.reshape(x_train, [x_train.shape[0], 3, 32, 32]),
                              np.reshape(x_test, [x_test.shape[0], 3, 32, 32]),
                              np.reshape(x_valid, [x_valid.shape[0], 3, 32, 32]))

  train_loader = torch.utils.data.DataLoader(Dataset(x_train, y_train),
                                             batch_size=batch_size)
  test_loader = torch.utils.data.DataLoader(Dataset(x_test, y_test),
                                            batch_size=batch_size)
  valid_loader = torch.utils.data.DataLoader(Dataset(x_valid, y_valid),
                                             batch_size=batch_size)

  ######################################################
  n_replicas = len(noise_list)
  models = []
  n_devices = torch.cuda.device_count()

  for dropout_rate in noise_list:
    models.append(Model(dropout_rate=dropout_rate))
    models[-1].to(DEVICE)

  optimizers = []
  for model in models:
    optimizers.append(torch.optim.SGD(model.parameters(), lr=learning_rate))
  summary = Summary(name, n_replicas)
  summary.initial_weight_vals = {}
  summary.initial_weight_vals = {i:concatinate_weights(model.parameters())
                                 for i, model in enumerate(models)}
  

  losses = []
  for model in models:
    losses.append(torch.nn.CrossEntropyLoss(reduction='mean').to(DEVICE))

  ####################################################################
  
  summary.curr_noise_vals = {i:n for i, n in enumerate(noise_list)}
  swap_attempts = 0
  swap_successes = 0
  accept_ratio = 0
  ####################################################################
  step = 0
  batch_logs = {'loss':{i:[] for i in range(n_replicas)},
                'error':{i:[] for i in range(n_replicas)}}
  try:
    for epoch in range(n_epochs):

      for (x, y) in train_loader:
      
        x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
        x, y = x.to(DEVICE), y.to(DEVICE)

        step += 1
        for i, model, optimizer, loss in zip(range(n_replicas), models, optimizers, losses):
          model.train()
          y_pred = model(x)
          
          loss_val = loss(y_pred.type(torch.FloatTensor), y.type(torch.LongTensor))
          _, y_pred = torch.max(y_pred, 1)
          err_val = compute_zero_one_error(y_pred, y)
          batch_logs['loss'][i].append(loss_val.item())
          batch_logs['error'][i].append(err_val)

          optimizer.zero_grad()
          loss_val.backward()
          optimizer.step()

        # test
        if step == 1 or step % test_step == 0:
          for i, model, loss in zip(range(n_replicas), models, losses):
            
            model.eval()
            test_loss, test_error = run_test_epoch(model, loss, test_loader)

            summary._test_loss[i].append(test_loss)
            summary._test_err[i].append(test_error)
            model.train()
          summary._test_steps.append(step)
          test_errs = [summary._test_err[i][-1] for i in range(n_replicas)]


          msg = {'epoch': epoch + 1,
                 'step':step,
                 'accept': accept_ratio}
          msg.update({'l'+str(i):test_errs[i] for i in range(n_replicas)})
          print_log(msg)

        # validation and swaps
        if step == 1 or step % swap_step == 0:

          for i, model, loss in zip(range(n_replicas), models, losses):
            model.eval()

            test_loss, test_error = run_test_epoch(model, loss, valid_loader)
            summary._valid_loss[i].append(test_loss)
            summary._valid_err[i].append(test_error)

            model.train()

          if step >= burn_in_period and n_replicas > 1:
            candidate_to_swap = random.choice(list(range(n_replicas-1)))
            beta = [summary.curr_noise_vals[i] for i in range(n_replicas)]
            beta_id = [(b, i) for i, b in enumerate(beta)]
            beta_id.sort(key=lambda x: x[0])
            i = beta_id[candidate_to_swap][1]
            j = beta_id[candidate_to_swap+1][1]
            beta_i = beta_id[candidate_to_swap][0]
            beta_j = beta_id[candidate_to_swap+1][0]
            try:
              assert abs(noise_list.index(beta_i) - noise_list.index(beta_j)) == 1
            except AssertionError:
              print('The index of', beta_i, 'is', noise_list.index(beta_i),
                    'and index of', beta_j, 'is', noise_list.index(beta_j))
              print('The noise list is:\n', noise_list)
              raise
            li, lj = summary._valid_loss[i][-1], summary._valid_loss[j][-1]
            # (1-beta) is to be consistent with tf implementation
            proba = np.exp(proba_coeff*(li-lj)*((1.0 - beta_i) - (1.0 - beta_j)))
            swap_attempts += 1
            if np.random.uniform() < proba:
              summary.curr_noise_vals[i] = beta_j
              summary.curr_noise_vals[j] = beta_i
              swap_successes += 1
              accept_pair = [(i, 1), (j, 1)]
              for i, model in enumerate(models):
                model.dropout1.p = summary.curr_noise_vals[i]
                model.dropout2.p = summary.curr_noise_vals[i]
            else:
              accept_pair = [(i, 0), (j, 0)]

            accept_ratio = swap_successes/swap_attempts
            for p in accept_pair:
              summary._replica_accepts[p[0]].append(p[1])

            
          for i in range(n_replicas):
            summary._train_noise_vals[i].append(summary.curr_noise_vals[i])
          norms = []
          for i, model in enumerate(models):
            model.eval()
            norms.append(torch.dist(concatinate_weights(model.parameters()),
                                    summary.initial_weight_vals[i]))
            model.train()
          for i, norm in enumerate(norms):
            summary._diffusion_vals[i].append(norm.item())

          

          for i in range(n_replicas):
            summary._train_loss[i].append(np.mean(batch_logs['loss'][i]))
            summary._train_err[i].append(np.mean(batch_logs['error'][i]))
          summary._train_steps.append(step)
          summary._valid_steps.append(step)
          batch_logs = {'loss':{i:[] for i in range(n_replicas)},
                        'error':{i:[] for i in range(n_replicas)}}

        if epoch % 200 == 0:
          summary._epoch = epoch + 1
          summary.flush_summary()
  finally:
    summary._epoch = epoch + 1
    summary.flush_summary()








class Summary:
  def __init__(self, name, n_replicas, simulation_num=0):
    self._n_replicas = n_replicas
    self._name = name
    dirname = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    dirname = os.path.join(dirname, 'summaries')
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    self._dirname = os.path.join(dirname, name)
    if not os.path.exists(self._dirname):
      os.makedirs(self._dirname)

    filename = self._create_log_fname(simulation_num=simulation_num)
    self.n_replicas = n_replicas
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



    self._epoch = 0

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

  def get_dirname(self):
    """Returns a full path to the directory where the log is stored."""
    return self._dirname

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
        'accepts': self._replica_accepts,
        'latest_epoch': self._epoch,
    }

    if 'win' in sys.platform:
      with open(self._logfile_name, 'wb') as fo:
        pickle.dump(log_data, fo, protocol=pickle.HIGHEST_PROTOCOL)

    else:
      with open(self._logfile_name, 'wb', os.O_NONBLOCK) as fo:
        pickle.dump(log_data, fo, protocol=pickle.HIGHEST_PROTOCOL)


class Dataset(dataset.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __getitem__(self, index):
        return (self.data[index], self.labels[index])
    
    def __len__(self):
        return self.data.shape[0]



def print_log(dict_):
  #buff = json.dumps(dict_)
  """Prints train log to stdout with the beginning of the line character."""
  buff = '|'.join(['['+str(k)+ ':' +"{0:.3f}".format(v)+']' for k, v in sorted(dict_.items())])
  sys.stdout.write('\r' + buff)
  sys.stdout.flush()
  #print(dict_, flush=True)

def concatinate_weights(params):
  flatten = [p.view(p.size(0), -1) for p in params]
  flatten = sorted(flatten, key=lambda x: np.prod(list(x.size())))
  flatten = [p.view(np.prod(list(p.size())), 1) for p in flatten]
  concat = torch.cat(flatten, 0)
  return concat